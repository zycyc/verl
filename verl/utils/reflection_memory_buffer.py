# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Memory buffer for storing and retrieving question-answer pairs."""

import os
import json
import numpy as np
from typing import Dict, List, Optional, Tuple, Union
from sentence_transformers import SentenceTransformer
import torch
import faiss
import time


class ReflectionMemoryBuffer:
    """Memory buffer for storing and retrieving question-answer pairs."""

    def __init__(
        self,
        embedding_model: str = "all-MiniLM-L6-v2",
        similarity_threshold: float = 0.85,
        save_path: Optional[str] = None,
        load_path: Optional[str] = None,
    ):
        """Initialize the memory buffer.

        Args:
            embedding_model: Name of the sentence transformer model to use for embeddings
            similarity_threshold: Threshold for considering questions similar
            save_path: Path to save the memory buffer
            load_path: Path to load the memory buffer from
        """
        self.embedding_model = SentenceTransformer(embedding_model)
        self.similarity_threshold = similarity_threshold
        self.save_path = save_path

        # Initialize memory storage
        self.questions = []
        self.answers = []
        self.question_embeddings = None
        self.index = None

        # Load existing memory if provided
        if load_path and os.path.exists(load_path):
            self.load(load_path)

        # Add this property to the class:
        self._needs_rebuild = True

    def add_memory(self, question: str, answer: str, max_size: int = 5000) -> None:
        """Add a question-answer pair to the memory buffer with size limit.

        Args:
            question: The question
            answer: The ground truth answer
            max_size: Maximum number of entries to keep
        """
        # Check if this question is already in the buffer (avoid duplicates)
        if question in self.questions:
            # Question already exists, update the corresponding answer
            idx = self.questions.index(question)
            old_answer = self.answers[idx]
            self.answers[idx] = answer
            # print(f"Updated answer for existing question in memory buffer: {question[:50]}...", old_answer[:50], "->", answer[:50])
            return

        # Check if we need to remove old entries
        if len(self.questions) >= max_size:
            # Remove oldest entries (first in, first out)
            self.questions = self.questions[-max_size + 1 :]
            self.answers = self.answers[-max_size + 1 :]

            # Rebuild the embeddings and index from scratch
            self._update_embeddings()
        else:
            # Add to memory
            # print(f"Adding question to memory buffer: {question[:50]}...")
            self.questions.append(question)
            self.answers.append(answer)

            # Update embeddings and index for just the new question
            self._update_embeddings_incremental(question)

    def _update_embeddings_incremental(self, new_question: str) -> None:
        """Update the embeddings and index incrementally for a new question."""
        try:
            # Generate embedding for just the new question
            with torch.no_grad():  # Prevent gradient tracking
                new_embedding = self.embedding_model.encode(
                    [new_question], convert_to_tensor=True
                )
                # Explicitly detach from computation graph and move to CPU
                new_embedding = new_embedding.detach().cpu()

            # Convert to numpy for FAISS
            new_embedding_np = new_embedding.numpy()

            # Normalize the new embedding
            faiss.normalize_L2(new_embedding_np)

            # If this is the first question, create the index
            if self.index is None:
                dimension = new_embedding_np.shape[1]
                self.index = faiss.IndexFlatIP(
                    dimension
                )  # Inner product for cosine similarity
                self.question_embeddings = new_embedding_np
            else:
                # Otherwise, add to existing embeddings
                if self.question_embeddings is not None:
                    self.question_embeddings = np.vstack(
                        [self.question_embeddings, new_embedding_np]
                    )
                else:
                    self.question_embeddings = new_embedding_np

            # Add the new embedding to the index
            self.index.add(new_embedding_np)

            # Explicitly clear any remaining GPU memory
            del new_embedding
            torch.cuda.empty_cache()

        except Exception as e:
            print(f"Error in _update_embeddings_incremental: {e}")
            import traceback

            traceback.print_exc()

    def _update_embeddings(self) -> None:
        """Update the embeddings and index for all questions. Used during initialization."""
        if not self.questions:
            self.index = None
            self.question_embeddings = None
            return

        # Generate embeddings for all questions
        self.question_embeddings = (
            self.embedding_model.encode(self.questions, convert_to_tensor=True)
            .cpu()
            .numpy()
        )

        # Create or update FAISS index
        dimension = self.question_embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dimension)  # Inner product for cosine similarity

        # Normalize for cosine similarity
        faiss.normalize_L2(self.question_embeddings)

        # Add all embeddings to the index
        self.index.add(self.question_embeddings)

    def retrieve_memory(
        self, question: str, top_k: int = 1
    ) -> Optional[Tuple[str, str, float]]:
        """Retrieve the most similar question-answer pair from memory."""
        if not self.questions:
            return None

        # Rebuild index if needed before retrieval
        if (
            hasattr(self, "_needs_rebuild")
            and self._needs_rebuild
            and len(self.questions) > 0
        ):
            try:
                print(
                    f"Rebuilding index for {len(self.questions)} questions before retrieval"
                )
                start_time = time.time()
                self._update_embeddings()
                end_time = time.time()
                print(
                    f"Index rebuilding completed in {end_time - start_time:.2f} seconds"
                )
                self._needs_rebuild = False
            except Exception as e:
                print(f"Error rebuilding index before retrieval: {str(e)}")
                # If we can't build the index, try direct string matching as fallback
                for i, q in enumerate(self.questions):
                    if question.strip() == q.strip():
                        return (q, self.answers[i], 1.0)
                return None

        # If we have no index (and rebuild didn't help), just return None
        if self.index is None:
            return None

        # Rest of the method remains the same...
        try:
            # Encode the query
            query_embedding = self.embedding_model.encode(
                question, convert_to_tensor=True
            )
            query_embedding_np = query_embedding.cpu().numpy().reshape(1, -1)
            faiss.normalize_L2(query_embedding_np)

            # Search for similar questions
            similarities, indices = self.index.search(
                query_embedding_np, min(top_k, len(self.questions))
            )

            # Check if the best match exceeds the threshold
            if similarities[0][0] >= self.similarity_threshold:
                best_idx = indices[0][0]
                return (
                    self.questions[best_idx],
                    self.answers[best_idx],
                    similarities[0][0],
                )

            return None
        except Exception as e:
            print(f"Error in retrieval: {str(e)}")
            return None

    def save(self, path: Optional[str] = None) -> None:
        """Save the memory buffer to disk.

        Args:
            path: Path to save the memory buffer, defaults to self.save_path
        """
        save_path = path or self.save_path
        if not save_path:
            raise ValueError("No save path specified")

        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(save_path), exist_ok=True)

        # Save the memory data
        memory_data = {"questions": self.questions, "answers": self.answers}

        with open(save_path, "w") as f:
            json.dump(memory_data, f)

    def load(self, path: Optional[str] = None) -> None:
        """Load the memory buffer from disk.

        Args:
            path: Path to load the memory buffer from, defaults to self.save_path
        """
        load_path = path or self.save_path
        if not load_path:
            raise ValueError("No load path specified")

        with open(load_path, "r") as f:
            memory_data = json.load(f)

        self.questions = memory_data["questions"]
        self.answers = memory_data["answers"]

        # Update embeddings and index
        if self.questions:
            self._update_embeddings()

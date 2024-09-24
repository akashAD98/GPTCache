

##  2024-09-24 20:34:18,629 - 123258616988736 - adapter.py-adapter:281 - WARNING: failed to save the data to cache, error: Schema mismatch: Expected list of float32 vectors
from typing import List, Optional
import os
import lancedb
import numpy as np
import pyarrow as pa
import lancedb
from gptcache.manager.vector_data.base import VectorBase, VectorData
from gptcache.utils import import_lancedb, import_torch
import_torch()
import_lancedb()
class LanceDB(VectorBase):
    def __init__(self, persist_directory: Optional[str] = "/tmp/lancedb", table_name: str = "gptcache", top_k: int = 1):
        if persist_directory is None:
            persist_directory = "/tmp/lancedb"
        self._persist_directory = os.path.expanduser(persist_directory)  # Expand directory path if necessary
        self._table_name = table_name
        self._top_k = top_k
        # Initialize LanceDB database
        self._db = lancedb.connect(self._persist_directory)
        # Initialize or open table
        if self._table_name not in self._db.table_names():
            # If the table doesn't exist, create it with a basic schema
            schema = pa.schema([
                pa.field("id", pa.string()),
                pa.field("vector", pa.list_(pa.float32(),768))
            ])
            self._table = self._db.create_table(self._table_name, schema=schema)
        else:
            self._table = self._db.open_table(self._table_name)
    def mul_add(self, datas: List[VectorData], **kwargs):
        """Add multiple vectors to the LanceDB table"""
        # Extract vectors and their IDs from the input data
        vectors, vector_ids = map(list, zip(*((data.data.tolist(), str(data.id)) for data in datas)))
        # Get the dimensionality of the first vector (optional for dynamic lists)
        vector_dim = len(vectors[0]) if vectors else 0
        # If the table is not initialized, create it with the correct schema
        if self._table is None:
            # Define the schema without specifying list_size
            schema = pa.schema([
                pa.field("id", pa.string()),
                pa.field("vector", pa.list_(pa.float32(),768))  # Remove list_size for flexibility
            ])
            self._table = self._db.create_table(self._table_name, schema=schema)
        else:
            # If the table already exists, check if the vector dimensions match
            existing_schema = self._table.schema
            existing_vector_field = existing_schema.field("vector")
            if not pa.types.is_list(existing_vector_field.type) or not pa.types.is_float32(existing_vector_field.type.value_type):
                raise ValueError("Schema mismatch: Expected list of float32 vectors")
        # Add the data to the table
        self._table.add(({"id": vector_id, "vector": vector} for vector_id, vector in zip(vector_ids, vectors)))
    def search(self, data: np.ndarray, top_k: int = -1, **kwargs):
        """Search for the most similar vectors in the LanceDB table"""
        if self._table is None:
            raise ValueError("The table does not exist. Please initialize or create the table first.")
        if len(self._table) == 0:
            return []
        if top_k == -1:
            top_k = self._top_k
        results = self._table.search(data.tolist()).limit(top_k).to_list()
        return [(result["_distance"], int(result["id"])) for result in results]
    def delete(self, ids: List[int]):
        """Delete vectors from the LanceDB table based on IDs"""
        for vector_id in ids:
            self._table.delete(f"id = '{vector_id}'")
    def rebuild(self, ids: Optional[List[int]] = None):
        """Rebuild the index, if applicable"""
        return True
    def count(self):
        """Return the total number of vectors in the table"""
        return len(self._table)







## code 2 embedding erro 
## 138196416103488 - adapter.py-adapter:281 - WARNING: failed to save the data to cache, error: LanceError(Arrow): C Data interface error: Invalid

# from typing import List, Optional
# import os
# import lancedb
# import numpy as np
# import pyarrow as pa
# import lancedb
# from gptcache.manager.vector_data.base import VectorBase, VectorData
# from gptcache.utils import import_lancedb, import_torch

# import_torch()
# import_lancedb()

# class LanceDB(VectorBase):
#     def __init__(self, persist_directory: Optional[str] = "/tmp/lancedb", table_name: str = "gptcache", top_k: int = 1):
#         if persist_directory is None:
#             persist_directory = "/tmp/lancedb"
#         self._persist_directory = os.path.expanduser(persist_directory)
#         self._table_name = table_name
#         self._top_k = top_k

#         # Initialize LanceDB database
#         self._db = lancedb.connect(self._persist_directory)

#         # Initialize or open table
#         if self._table_name not in self._db.table_names():
#             # If the table doesn't exist, create it with the corrected schema
#             schema = pa.schema([
#                 pa.field("id", pa.string()),
#                 pa.field("vector", pa.list_(pa.float32()))
#             ])
#             self._table = self._db.create_table(self._table_name, schema=schema)
#         else:
#             self._table = self._db.open_table(self._table_name)

#     def mul_add(self, datas: List[VectorData], **kwargs):
#         """Add multiple vectors to the LanceDB table"""
#         # Extract vectors and their IDs from the input data
#         vectors, vector_ids = map(list, zip(*((data.data.tolist(), str(data.id)) for data in datas)))

#         # Add the data to the table
#         self._table.add(({"id": vector_id, "vector": vector} for vector_id, vector in zip(vector_ids, vectors)))

#     def search(self, data: np.ndarray, top_k: int = -1, **kwargs):
#         """Search for the most similar vectors in the LanceDB table"""
#         if self._table is None:
#             raise ValueError("The table does not exist. Please initialize or create the table first.")
#         if len(self._table) == 0:
#             return []
#         if top_k == -1:
#             top_k = self._top_k
#         results = self._table.search(data.tolist()).limit(top_k).to_list()
#         return [(result["_distance"], int(result["id"])) for result in results]

#     def delete(self, ids: List[int]):
#         """Delete vectors from the LanceDB table based on IDs"""
#         for vector_id in ids:
#             self._table.delete(f"id = '{vector_id}'")

#     def rebuild(self, ids: Optional[List[int]] = None):
#         """Rebuild the index, if applicable"""
#         return True

#     def count(self):
#         """Return the total number of vectors in the table"""
#         return len(self._table)

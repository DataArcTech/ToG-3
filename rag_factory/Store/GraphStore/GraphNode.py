from typing import Optional, List, Dict, Any, Tuple
from pydantic import BaseModel, Field
from abc import abstractmethod, ABC
import fsspec

class Node(BaseModel):
    """Base node class for the graph."""
    label: str = Field(default="node", description="The label of the node.(entity type)")
    embedding: Optional[List[float]] = Field(default=None, description="The embedding of the node.")
    metadatas: Optional[Dict[str, Any]] = Field(default_factory=dict, description="The metadata of the node.")

    @abstractmethod
    def __str__(self) -> str:
        """Return the string representation of the node."""
        ...

    @property
    @abstractmethod
    def id(self) -> str:
        """Get the node id."""
        ...


class EntityNode(Node):
    """Entity node class for the graph."""
    name: str = Field(description="The name of the entity.")
    label: str = Field(default="entity", description="The label of the node.(entity type)")
    metadatas: Optional[Dict[str, Any]] = Field(default_factory=dict, description="The metadata of the node.")

    def __str__(self) -> str:
        """Return the string representation of the node."""
        if self.metadatas:
            return f"{self.name} ({self.metadatas})"
        return self.name

    @property
    def id(self) -> str:
        """Get the node id."""
        return self.name


# TODO id hash? 哈希去重
class DocumentNode(Node):
    """Document node class for the graph."""
    content: str = Field(description="The content of the document.")
    id_: Optional[str] = Field(
        default=None, description="The id of the node. Defaults to a hash of the text."
    )
    label: str = Field(default="text_chunk", description="The label of the node.")
    metadatas: Optional[Dict[str, Any]] = Field(default_factory=dict, description="The metadata of the node.")

    def __str__(self) -> str:
        return self.content
    
    def id(self) -> str:
        """Get the DocumentNode id."""
        return str(hash(self.content)) if self.id_ is None else self.id_


class Relation(BaseModel):
    """Relation class for the graph. A Relation is a connection between two entities."""
    label: str = Field(description="The label of the relation.")
    head_id: str = Field(description="The id(name) of the head node.")
    tail_id: str = Field(description="The id(name) of the tail node.")
    metadatas: Optional[Dict[str, Any]] = Field(default_factory=dict, description="The metadata of the relation.")

    def __str__(self) -> str:
        """Return the string representation of the relation."""
        if self.metadatas:
            return f"{self.label} ({self.metadatas})"
        return self.label

    @property
    def id(self) -> str:
        """Get the relation id."""
        return self.label
    

Triplet = Tuple[EntityNode, Relation, EntityNode]

class GraphStore(ABC):
    """
    Abstract graph store protocol.

    Attributes:
        client: Any: The client used to connect to the graph store.
        get: Callable[[str], List[List[str]]]: Get triplets for a given subject.
        get_rel_map: Callable[[Optional[List[str]], int], Dict[str, List[List[str]]]]:
            Get subjects' rel map in max depth.
        upsert_triplet: Callable[[str, str, str], None]: Upsert a triplet.
        delete: Callable[[str, str, str], None]: Delete a triplet.
        persist: Callable[[str, Optional[fsspec.AbstractFileSystem]], None]:
            Persist the graph store to a file.
    """

    supports_structured_queries: bool = False
    supports_vector_queries: bool = False
    # text_to_cypher_template: PromptTemplate = DEFAULT_CYPHER_TEMPALTE

    @property
    def client(self) -> Any:
        """Get client."""
        ...
    
    @abstractmethod
    def get(self, metadata: Optional[Dict[str, Any]] = None, ids: Optional[List[str]] = None) -> List[Node]:
        """Get nodes with matching values."""
        ...
    
    @abstractmethod
    def get_triplets(self, entity_names: Optional[List[str]] = None, relation_names: Optional[List[str]] = None, metadata: Optional[Dict[str, Any]] = None, ids: Optional[List[str]] = None) -> List[Triplet]:
        """Get triplets with matching values."""
        ...

    @abstractmethod
    def get_rel_map(self, nodes: List[Node], depth: int = 2, limit: int = 30, ignore_rels: Optional[List[str]] = None) -> List[Triplet]:
        """Get depth-aware rel map."""
        ...

    @abstractmethod
    def upsert_nodes(self, nodes: List[Node]) -> None:
        """Upsert nodes."""
        ...
    
    @abstractmethod
    def upsert_relations(self, relations: List[Relation]) -> None:
        """Upsert relations."""
        ...

    @abstractmethod
    def delete(self, entity_names: Optional[List[str]] = None, relation_names: Optional[List[str]] = None, metadata: Optional[Dict[str, Any]] = None, ids: Optional[List[str]] = None) -> None:
        """Delete nodes and relations."""
        ...
    
    @abstractmethod
    def structured_query(self, query: str, param_map: Optional[Dict[str, Any]] = None) -> Any:
        """Query the graph store with a query."""
        ...

    # @abstractmethod
    # def vector_query(self, query: VectorStoreQuery, **kwargs: Any) -> Tuple[List[Node], List[float]]:
    #     """Query the graph store with a vector store query."""
    #     ...
    
    @abstractmethod
    def persist(self, persist_path: str, fs: Optional[fsspec.AbstractFileSystem] = None) -> None:
        """Persist the graph store to a file."""
        ...

    @abstractmethod
    def get_schema(self, refresh: bool = False) -> str:
        """Get the schema of the graph store."""
        ...

    def get_schema_str(self, refresh: bool = False) -> str:
        """Get the schema of the graph store as a string."""
        return str(self.get_schema(refresh=refresh))

    ### ----- Async Methods ----- ###

    async def aget(
        self,
        metadata: Optional[Dict[str, Any]] = None,
        ids: Optional[List[str]] = None,
    ) -> List[Node]:
        """Get nodes with matching values."""
        return self.get(metadata, ids)
    
    async def aget_triplets(
        self,
        entity_names: Optional[List[str]] = None,
        relation_names: Optional[List[str]] = None,
        metadata: Optional[Dict[str, Any]] = None,
        ids: Optional[List[str]] = None,
    ) -> List[Triplet]:
        """Get triplets with matching values."""
        return self.get_triplets(entity_names, relation_names, metadata, ids)

    async def aget_rel_map(
        self,
        nodes: List[Node],
        depth: int = 2,
        limit: int = 30,
        ignore_rels: Optional[List[str]] = None,
    ) -> List[Triplet]:
        """Get depth-aware rel map."""
        return self.get_rel_map(nodes, depth, limit, ignore_rels)

    async def aupsert_nodes(self, nodes: List[Node]) -> None:
        """Upsert nodes."""
        return self.upsert_nodes(nodes)
    
    async def aupsert_relations(self, relations: List[Relation]) -> None:
        """Upsert relations."""
        return self.upsert_relations(relations)
    
    async def adelete(
        self, 
        entity_names: Optional[List[str]] = None, 
        relation_names: Optional[List[str]] = None, 
        metadata: Optional[Dict[str, Any]] = None, 
        ids: Optional[List[str]] = None
    ) -> None:
        """Delete nodes and relations."""
        return self.delete(entity_names, relation_names, metadata, ids)
    
    async def astructured_query(
        self,
        query: str,
        param_map: Optional[Dict[str, Any]] = None
    ) -> Any:
        """Query the graph store with a structured query."""
        return self.structured_query(query, param_map)
    
    # async def avector_query(self, query: VectorStoreQuery, **kwargs: Any) -> Tuple[List[Node], List[float]]:
    #     """Query the graph store with a vector store query."""
    #     return self.vector_query(query, **kwargs)
    
    async def aget_schema(self, refresh: bool = False) -> str:
        """Get the schema of the graph store."""
        return self.get_schema(refresh=refresh)
    
    async def aget_schema_str(self, refresh: bool = False) -> str:
        """Get the schema of the graph store as a string."""
        return str(await self.aget_schema(refresh=refresh))
from typing import Optional, List, Dict, Any, Tuple
from pydantic import BaseModel, Field
from abc import abstractmethod
from datetime import datetime

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


class ChunkNode(Node):
    """Chunk node class for storing original text fragments."""
    id_: str = Field(description="Unique identifier for the chunk (Chunk_xxx)")
    content: str = Field(description="Original text content")
    source: str = Field(description="Document/file source")
    create_time: datetime = Field(default_factory=datetime.now, description="Creation time")
    event: List[str] = Field(default_factory=list, description="List of event IDs contained in this chunk")
    mention: List[str] = Field(default_factory=list, description="List of mention IDs directly extracted")
    label: str = Field(default="Chunk", description="Node label")

    def __str__(self) -> str:
        """Return the string representation of the chunk."""
        return f"Chunk({self.id_}): {self.content[:100]}..."

    @property
    def id(self) -> str:
        """Get the chunk id."""
        return self.id_

    def add_event(self, event_id: str) -> None:
        """Add an event ID to this chunk."""
        if event_id not in self.event:
            self.event.append(event_id)

    def add_mention(self, mention_id: str) -> None:
        """Add a mention ID to this chunk."""
        if mention_id not in self.mention:
            self.mention.append(mention_id)




class EventNode(Node):
    """Event node class for storing extracted events."""
    id_: str = Field(description="Unique identifier for the event (Event_xxx)")
    content: str = Field(description="Event text content")
    source: str = Field(description="Source chunk ID")
    create_time: datetime = Field(default_factory=datetime.now, description="Creation time")
    update_time: datetime = Field(default_factory=datetime.now, description="Last update time")
    mention: List[str] = Field(default_factory=list, description="List of mention IDs involved in this event")
    label: str = Field(default="Event", description="Node label")

    def __str__(self) -> str:
        """Return the string representation of the event."""
        return f"Event({self.id_}): {self.content[:100]}..."

    @property
    def id(self) -> str:
        """Get the event id."""
        return self.id_

    def add_mention(self, mention_id: str) -> None:
        """Add a mention ID to this event."""
        if mention_id not in self.mention:
            self.mention.append(mention_id)
            self.update_time = datetime.now()

    def update_content(self, content: str) -> None:
        """Update event content and timestamp."""
        self.content = content
        self.update_time = datetime.now()



class MentionNode(Node):
    """Mention node class for storing entity mentions in original text."""
    id_: str = Field(description="Unique identifier for the mention (Mention_xxx)")
    text: str = Field(description="Mention text from original text")
    name: str = Field(description="Normalized Mention name")
    description: str = Field(description="Mention description")
    source: str = Field(description="Source chunk ID")
    event: Optional[str] = Field(default=None, description="Event ID this mention belongs to")

    def __str__(self) -> str:
        """Return the string representation of the mention."""
        return f"Mention({self.id_}): {self.text}"

    @property
    def id(self) -> str:
        """Get the mention id."""
        return self.id_

    def set_event_reference(self, event_id: str) -> None:
        """Set the event reference."""
        self.event = event_id




class EntityNode(Node):
    """Entity node class for storing disambiguated final entities."""
    id_: str = Field(description="Unique identifier for the entity (Entity_xxx)")
    name: str = Field(description="The name of the entity")
    aliases: Optional[List[str]] = Field(default=None, description="Aliases and synonyms")
    description: Dict[str, str] = Field(default_factory=dict, description="Source(mention_id) to description mapping")
    summary: Optional[str] = Field(default=None, description="System generated summary")
    create_time: datetime = Field(default_factory=datetime.now, description="Creation time")
    update_time: datetime = Field(default_factory=datetime.now, description="Last update time")

    def __str__(self) -> str:
        """Return the string representation of the entity."""
        aliases_info = f" (aliases: {', '.join(self.aliases)})" if self.aliases else ""
        return f"Entity({self.id_}): {self.name}{aliases_info}"

    @property
    def id(self) -> str:
        """Get the entity id."""
        return self.id_

    def add_alias(self, alias: str) -> None:
        """Add an alias to this entity."""
        if alias not in self.aliases:
            self.aliases.append(alias)
            self.update_time = datetime.now()

    def add_description(self, source: str, description: str) -> None:
        """Add a description from a specific source."""
        self.description[source] = description
        self.update_time = datetime.now()

    def update_summary(self, summary: str) -> None:
        """Update the system generated summary."""
        self.summary = summary
        self.update_time = datetime.now()




class Relation(BaseModel):
    """Relation class for the graph. A Relation is a connection between two nodes."""
    label: str = Field(description="The label of the relation.")
    head_id: str = Field(description="The id of the head node.")
    tail_id: str = Field(description="The id of the tail node.")
    relation_type: Optional[str] = Field(default=None, description="Specific type of relation (for semantic relations)")
    metadatas: Optional[Dict[str, Any]] = Field(default_factory=dict, description="The metadata of the relation.")

    def __str__(self) -> str:
        """Return the string representation of the relation."""
        type_info = f":{self.relation_type}" if self.relation_type else ""
        metadata_info = f" ({self.metadatas})" if self.metadatas else ""
        return f"{self.label}{type_info}{metadata_info}"

    @property
    def id(self) -> str:
        """Get the relation id."""
        return f"{self.head_id}_{self.label}_{self.tail_id}"



Triplet = Tuple[EntityNode, Relation, EntityNode]

class Entity:
    def __init__(self, name, type, description):
        self.name:str = name
        self.type:str = type
        self.description:str = description  # 实体描述
    
    def __eq__(self, other):
        return self.name == other.name

    def __hash__(self):
        return hash(self.name)

    def __repr__(self):
        return f"Entity({self.name}, {self.type}, {self.description})"

class Position:
    '''
    位置类，表示一个实体的位置，包括文章id，段id和句子id
    '''
    def __init__(self):
        self.article_id = None
        self.paragraph_id = None
        self.sentence_id = None

    @staticmethod
    def form_id(article_id, paragraph_id, sentence_id):
        position = Position()
        position.article_id = article_id
        position.paragraph_id = paragraph_id
        position.sentence_id = sentence_id
        return position

    def to_dict(self):
        return {
            "art_id":self.article_id,
            "par_id":self.paragraph_id,
            "sen_id":self.sentence_id
        }

    @staticmethod
    def from_dict(dict):
        position = Position()
        position.sentence_id = dict["sen_id"]
        position.paragraph_id = dict["par_id"]
        position.article_id = dict["art_id"]
        return position

    def __eq__(self, other):
        return self.article_id == other.article_id and self.paragraph_id == other.paragraph_id and self.sentence_id == other.sentence_id

    def __repr__(self):
        return f"Position({self.article_id}, {self.paragraph_id}, {self.sentence_id})"

    def __hash__(self):
        return hash((self.article_id, self.paragraph_id, self.sentence_id))

class Node:
    def __init__(self, name=None):
        self.name = name
        self.id = None  # 初始时不分配 ID
        self.type = set() 
        self.description = []
        self.positions = set()
        self.sen_childs = set() 
        self.par_childs = set()
        self.art_childs = set()

    @staticmethod
    def reset_id():
        Node.id = 0

    @staticmethod
    def form_entity(entity: Entity):
        node = Node(name=entity.name)
        node.type.add(entity.type)
        # node.description.append(entity.description)
        return node
    
    @staticmethod
    def from_dict(dict):
        node = Node()
        node.name = dict["name"]
        node.id = dict["id"]
        node.type = set(dict["type"])
        node.description = dict["description"]
        node.positions = set([Position.from_dict(position) for position in dict["positions"]])
        node.sen_childs = set(dict["sen_childs"])
        node.par_childs = set(dict["par_childs"])
        node.art_childs = set(dict["art_childs"])
        return node

    def to_dict(self):
        return {
            "name":self.name,
            "id":self.id,
            "type":list(self.type),
            "description":self.description,
            "positions":[position.to_dict() for position in self.positions],
            "sen_childs":list(self.sen_childs),
            "par_childs":list(self.par_childs),
            "art_childs":list(self.art_childs),
        }

    def refresh_sen_child(self, entitys):
        for entity in entitys:
            if entity.name == self.name:
                continue
            self.sen_childs.add(entity.name)
            self.par_childs.discard(entity.name) #时间先后的问题
            self.art_childs.discard(entity.name)
    
    def refresh_par_child(self, entitys): 
        for entity in entitys:
            if entity.name == self.name or entity.name in self.sen_childs:
                continue
            self.par_childs.add(entity.name)
            self.art_childs.discard(entity.name)

    def refresh_art_child(self, entitys):
        for entity in entitys:
            if entity.name == self.name or entity.name in self.sen_childs or entity.name in self.par_childs:
                continue
            self.art_childs.add(entity.name)
    
    def refresh_position(self, position):
        self.positions.add(position)    
    
    def refresh_description(self, description):
        self.description.append(description)
    
    def __repr__(self) -> str:
        return (
            f"Node(ID: {self.id}, Name: {self.name}, Type: {', '.join(self.type)}, "
            f"Description: {', '.join(self.description)}, "
            f"Positions: {', '.join(str(pos) for pos in self.positions)}, "
            f"Sen_Childs: {', '.join(self.sen_childs)}, "
            f"Par_Childs: {', '.join(self.par_childs)}, "
            f"Art_Childs: {', '.join(self.art_childs)})"
        )


import json
class Corpus:
    def __init__(self, corpus_file : list[str]):
        
        # article_path = []
        # with open(corpus_file, 'r') as f:
        #     for i, line in enumerate(f):
        #         article = json.loads(line)
        #         article_path.append(article["source_file"])
        total_corpus = []
        for file in corpus_file:
            corpus_data = []
            with open(file, 'r') as f:
                for line in f:
                    article = json.loads(line)
                    corpus_data.append(article["chunks"])
            total_corpus.append(corpus_data)
        self.corpus = total_corpus

    def idx_to_content(self, art_id, par_id, sen_id):
        return self.corpus[int(art_id)][int(par_id)][int(sen_id)]

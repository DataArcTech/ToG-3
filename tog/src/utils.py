import torch  
from transformers import AutoTokenizer, AutoModel  
from chromadb import PersistentClient  
from openai import OpenAI  
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
from concurrent.futures import ProcessPoolExecutor, as_completed, ThreadPoolExecutor

# 配置LLM客户端
def get_llm_client(base_url, api_key):
    return OpenAI(
        base_url=base_url,
        api_key=api_key
    )

# 配置Chroma客户端
def get_chroma_client(db_path):
    return PersistentClient(path=db_path)

def load_model_and_tokenizer(model_path):  
    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModel.from_pretrained(model_path).to('cuda')
    return model, tokenizer 


def get_embeddings(query, model, tokenizer):
    '''
    计算embedding
    '''
    inputs = tokenizer(query, return_tensors='pt', padding=True, truncation=True).to('cuda')
    with torch.no_grad():
        embeddings = model(**inputs).last_hidden_state.mean(dim=1).cpu().numpy()
    return embeddings


# 构建过滤ID(定向检索)
def build_ids_to_filter(topic_entities, graph_info):  
    '''  
    给query中的每个实体构建过滤的ID，从这里面进行检索。  
    返回一个字典：  
    {"实体1":[位置1, 位置2, ...], "实体2":[位置1, 位置2, ...]}  
    这里不需要名字匹配，只需要顺序一致即可。  
    '''  
    ids_to_filter_dict = {}  
  
    # 遍历每个topic实体的索引  
    for index, entity_name in enumerate(topic_entities):  
        ids_to_filter_dict[entity_name] = []  # 为每个实体初始化一个空列表来存储位置
  
        # 尝试从graph_info中对应索引的位置获取实体信息  
        if index < len(graph_info) and graph_info[index]:  
            # 遍历列表中的每个实体信息字典（尽管这里假设每个索引只对应一个实体信息）  
            for entity_info in graph_info[index]:  
                # 更新实体的位置信息（这里不再检查实体名称是否匹配）  
                ids_to_filter_dict[entity_name].extend(entity_info['positions'])  
  
    return ids_to_filter_dict  




def retrieve_from_graph(collection, query_embeddings, n_results, ids_to_filter=None):
    """
    从向量数据库中检索数据。

    Args:
        collection (ChromaCollection): 要检索的集合对象。
        query_embeddings (np.ndarray): 查询的嵌入表示。
        n_results (int): 要检索的结果数量。
        ids_to_filter (list, optional): 用于过滤结果的ID列表。默认为None。

    Returns:
        dict: 检索结果，或者部分结果，如果无法检索到足够多的结果。
    """
    # where_clause = {"position_p": {"$in": ids_to_filter}} if ids_to_filter else None # 段落检索
    where_clause = {"position": {"$in": ids_to_filter}} if ids_to_filter else None # 句子检索
    
    # 使用递减策略调整 n_results，确保不抛出异常
    while n_results > 0:
        try:
            # 尝试检索结果
            query_embeddings = query_embeddings
            results = collection.query(
                query_embeddings=query_embeddings,
                n_results=n_results,
                where=where_clause,
                include=["embeddings","documents", "distances","metadatas"]
            )

            return results

        except RuntimeError as e:
            # 发生错误时，减少 n_results 并再次尝试检索
            print(f"Warning: Error occurred during retrieval: {e}, reducing n_results to {n_results-1}")
            n_results -= 1
    
    # 如果检索到的结果为空，则返回空字典
    print("No results could be retrieved.")
    return {"ids": [], "distances": []}




def process_ids(codes):
    '''
    将ids_to_filter_dict转换为对应段落的positions
    '''
    # 去掉最后一位并去重
    processed_codes = list(set([code.rsplit('-', 1)[0] for code in codes]))
    # 对列表排序
    processed_codes.sort()
    return processed_codes

def process_ids_list(codes):
    '''
    输入为列表的情况下将ids_to_filter转换为对应段落的positions
    '''
    # 去掉最后一位并去重  
    processed_codes = tuple(set(code.rsplit('-', 1)[0] for code in codes))  # 直接生成元组  
    return processed_codes  
    

def get_graph_info(topic_entities_result):  
    '''  
    对topic_entity进行图构建,构建的图包括实体名称，边和位置信息以及边的权重 
    新增加了相关实体ID信息，和实体名称一起以元组形式保存
    ## 增加了归一化处理后的权重信息
       graph_info = [   
                      [{"name": "entity1", "positions": ['position1', 'position2' ...], "edges": [('entity_name1', 'entity_id1'), ('entity_name2', 'entity_id2') ...], "weight": [weight1, weight2 ...], "normalize_weight": [norm_weight1, norm_weight2 ...] },  
                       {"name": "entity2", "positions": ['position1', 'position2' ...], "edges": [('entity_name3', 'entity_id3'), ('entity_name4', 'entity_id4') ...], "weight": [weight3, weight4 ...], "normalize_weight": [norm_weight3, norm_weight4 ...] }],  
                      ...  
                     ]  
    '''  
    graph_info = []  
    # 从results的metadatas中提取出positions和edges的信息  
    # 解析每个查询实体的结果    
    for i, entity_results in enumerate(topic_entities_result["metadatas"]):    
        entity_list = []    
        for item in entity_results:    
            # 提取并格式化positions, edges, weights, 和 normalize_weights  
            edges_names = item['edges'].split('<delimiter>') if item['edges'] else []
            edges_ids = item['edges_id'].split('<delimiter>') if item['edges_id'] else []
            edges_normalize_weights = item['normlize_weight'].split('<delimiter>') if item['normlize_weight'] else []

            # 如果任何一个为空，则edges为空列表
            if not edges_names or not edges_ids or not edges_normalize_weights:
                edges = []
            else:
                edges = list(zip(edges_names, edges_ids, edges_normalize_weights))

            formatted_item = {    
                "name": item['name'],    
                "positions": item['positions'].split('<delimiter>') if item['positions'] else [],    
                "edges": edges,  
                "weight": [float(w) for w in item['weight'].split('<delimiter>') if w] if item['weight'] else [],  
                "normlize_weight": [float(w) for w in item['normlize_weight'].split('<delimiter>') if w] if item['normlize_weight'] else []
            }    

            entity_list.append(formatted_item)    
        graph_info.append(entity_list)  

    return graph_info

def get_graph_info(topic_entities_result, filter_result):
    """
    构建实体的图信息，包含筛选后的实体。
    
    Parameters:
        topic_entities_result (dict): 包含原始实体信息的数据。
        filter_result (list): 筛选结果，格式为 {"topic_entity": [(name, entity_id, weight), (name, entity_id, weight), ...]}。

    Returns:
        graph_info (list): 包含筛选后实体图信息的列表。
    """
    graph_info = []
    # 将 filter_result 转换为便于查找的集合
    filter_set = { (name, entity_id) for group in filter_result for name, entity_id, _ in group }

    # 遍历每个实体组
    for i, entity_results in enumerate(topic_entities_result["metadatas"]):
        entity_list = []

        for item in entity_results:
            entity_name = item['name']
            entity_id = item['entity_id']
            
            # 检查实体是否在筛选结果中
            if (entity_name, entity_id) not in filter_set:
                continue  # 跳过未通过筛选的实体

            # 提取并格式化 positions, edges, weights, 和 normalize_weights
            edges_names = item['edges'].split('<delimiter>') if item['edges'] else []
            edges_ids = item['edges_id'].split('<delimiter>') if item['edges_id'] else []
            edges_normalize_weights = item['normlize_weight'].split('<delimiter>') if item['normlize_weight'] else []

            # 如果任何一个为空，则 edges 为空列表
            if not edges_names or not edges_ids or not edges_normalize_weights:
                edges = []
            else:
                edges = list(zip(edges_names, edges_ids, edges_normalize_weights))

            formatted_item = {
                "name": entity_name,
                "positions": item['positions'].split('<delimiter>') if item['positions'] else [],
                "edges": edges,
                "weight": [float(w) for w in item['weight'].split('<delimiter>') if w] if item['weight'] else [],
                "normlize_weight": [float(w) for w in item['normlize_weight'].split('<delimiter>') if w] if item['normlize_weight'] else []
            }

            entity_list.append(formatted_item)

        graph_info.append(entity_list)

    return graph_info


def get_graph_info_list(topic_entities_result, filter_result):
    """
    构建实体的图信息，包含筛选后的实体。
    
    Parameters:
        topic_entities_result (dict): 包含原始实体信息的数据。
        filter_result (list): 筛选结果，格式为 [(name, entity_id, weight), (name, entity_id, weight), ...]。

    Returns:
        graph_info (list): 包含筛选后实体图信息的列表。
    """
    graph_info = []
    # 将 filter_result 转换为便于查找的集合
    filter_set = { (name, entity_id) for name, entity_id, _ in filter_result }

    # 遍历每个实体组
    for i, entity_results in enumerate(topic_entities_result["metadatas"]):
        entity_list = []

        for item in entity_results:
            entity_name = item['name']
            entity_id = item['entity_id']
            
            # 检查实体是否在筛选结果中
            if (entity_name, entity_id) not in filter_set:
                continue  # 跳过未通过筛选的实体

            # 提取并格式化 positions, edges, weights, 和 normalize_weights
            edges_names = item['edges'].split('<delimiter>') if item['edges'] else []
            edges_ids = item['edges_id'].split('<delimiter>') if item['edges_id'] else []
            edges_normalize_weights = item['normlize_weight'].split('<delimiter>') if item['normlize_weight'] else []

            # 如果任何一个为空，则 edges 为空列表
            if not edges_names or not edges_ids or not edges_normalize_weights:
                edges = []
            else:
                edges = list(zip(edges_names, edges_ids, edges_normalize_weights))

            formatted_item = {
                "name": entity_name,
                "positions": item['positions'].split('<delimiter>') if item['positions'] else [],
                "edges": edges,
                "weight": [float(w) for w in item['weight'].split('<delimiter>') if w] if item['weight'] else [],
                "normlize_weight": [float(w) for w in item['normlize_weight'].split('<delimiter>') if w] if item['normlize_weight'] else []
            }

            entity_list.append(formatted_item)

        graph_info.append(entity_list)

    return graph_info




def concat_query_and_documents(topic_sentence_info):
    '''
    Concatenate the query and the retrieved documents (English version)
    '''
    retrieve_information = ""
    for entity, results in topic_sentence_info.items():
        if isinstance(results, list):
            documents = "<keyword>{}</keyword>:\n".format(entity)
            for i, doc in enumerate(results):
                documents += f"{i+1}. {doc['document']}\n\n"

            retrieve_information += documents + "\n\n"
        else:
            retrieve_information += f"{entity}:\n(Unrecognized document format)\n\n"
    
    return retrieve_information
    
def extract_info_from_llm_answer(answer):  
    # 这里应该实现具体的逻辑来判断答案是否足够  
    # 返回True表示足够，False表示不足够，None表示无法判断  
    # 示例逻辑（需要根据实际需求修改）：  
    if "[YES]" in answer:  
        return True  
    elif "[NO]" in answer:  
        return False  
    else:  
        return None 
    

def transform_graph_info_to_ids_filter(graph_info):
    result = {}  
    # 遍历metadatas列表  
    for metadata in graph_info['metadatas']:  
        for item in metadata:  
            # 获取实体名称和位置信息  
            entity_name = item['name']  
            positions = item['positions'].split(',')  
            
            # 如果实体名称不在结果字典中，则添加  
            if entity_name not in result:  
                result[entity_name] = []  
            
            # 添加位置信息到对应的实体列表中  
            result[entity_name].extend(positions)  
    return result

def transform_data(data):  
    # 初始化结果列表  
    result = []  
  
    # 遍历每个元素  
    for i in range(len(data['ids'])):  
        # 获取当前元素的metadata  
        metadata = data['metadatas'][i][0]  
          
        # 初始化一个新的字典来保存转换后的数据  
        transformed_entry = {  
            'name': metadata['name'],  
            'positions': metadata['positions'].split(','),  
            'edges': metadata['edges'].split(', '),  
            'weight': metadata['weight'].split(', '),  
            'normlize_weight': metadata['normlize_weight'].split(', ')  
        }  
          
        # 将转换后的字典添加到结果列表中  
        result.append([transformed_entry])  
      
    return result  

def extract_unique_positions(graph_info):  
    '''
    对graph_info中所有的实体，提取出一个集合positions，用于定向检索
    '''
    # 初始化一个空的set来存储所有唯一的positions  
    positions_set = set()  
  
    # 遍历graph_info中的每个实体组  
    for entity_group in graph_info:  
        # 遍历实体组中的每个实体  
        for entity in entity_group:  
            # 更新positions_set，将实体的positions添加进去  
            positions_set.update(entity["positions"])  
  
    # 将set转换为list  
    ids_to_filter = list(positions_set)  
  
    # 返回结果  
    return ids_to_filter 



def extract_documents(data):  
    # 检查数据中是否包含 'documents' 字段  
    if 'documents' in data and 'distances' in data:  
        # 提取 'documents' 字段的内容  
        documents = data['documents']  
        distances = data['distances']

        if not isinstance(documents, list):  
            documents = [documents]  
        if not isinstance(distances, list):  
            distances = [distances]

        result = list(zip(documents, distances))

        if len(result) > 0 and len(result[0]) == 2:  
            return result  
        else:  
            return []  # 如果数据不匹配，返回空列表  
    else:  
        # 如果 'documents' 或 'distances' 字段不存在，返回一个空列表  
        return [] 

def get_documents_from_dict(data):  
    '''
    将向量数据库的result字典转换为句子字典，包含句子和距离信息
    '''
    results = []  
      
    if 'distances' in data and 'documents' in data:  
        documents = data['documents']  
        distances = data['distances']  
        # 确保documents和distances的列表长度相匹配  
        for doc_list, dist_list in zip(documents, distances):  
            for document, distance in zip(doc_list, dist_list):  
                result = {'document': document, 'distance': distance}  
                results.append(result)  
    return results

def extract_documents_diff(data):  
    """  
    从嵌套字典中提取所有的documents信息。  
  
    参数:  
    data (dict): 包含嵌套结构的字典。  
  
    返回:  
    list: 包含所有documents信息的列表。  
    """  
    documents_list = []  
  
    # 遍历字典的顶层键  
    for key in data:  
        # 检查每个顶层键对应的值是否包含'documents'键  
        if 'documents' in data[key]:  
            # 如果包含，则将其添加到documents_list中  
            documents_list.extend(data[key]['documents'])  
  
    return documents_list  

def reconstruct_query(query, extract_answer, related_documents):
    input = "用户提供的实体列表：\n" 
    for entity in extract_answer:
        input = input + entity + ", "
    input = input + "用户提供的文档列表：\n"
    for doc_list in related_documents:
        for doc in doc_list:
            input = input + doc + "\n"
    input = input + "下面是用户的提问，请按照要求回复用户的问题：\n" + query 
    return input


# def get_related_entities(graph_info):  
#     """  
#     从嵌套的graph_info结构中提取所有的边，并去除重复项。  
#     返回('实体名', 'ID', 'weight')这样的元组
  
#     参数:  
#     graph_info (list of list of dict): 一个嵌套列表，其中包含字典，每个字典代表一个实体，并包含'edges'键。  
  
#     返回:  
#     list: 包含所有唯一边的列表。  
#     """  
#     edge_set = set()  # 使用集合来存储唯一的边  
#     for sublist in graph_info:  
#         for entity in sublist:  
#             if 'edges' in entity:  
#                 # 遍历edges中的每个元组，并将其添加到集合中  
#                 for edge in entity['edges']:  
#                     edge_set.add(edge)  
#     return list(edge_set)  # 将集合转换为列表并返回 

def get_related_entities(graph_info):  
    """  
    从嵌套的graph_info结构中提取所有的边，合并前两个元素相同的边的数值，并去除重复项。  
    返回('实体名', 'ID', 'weight')这样的元组
  
    参数:  
    graph_info (list of list of dict): 一个嵌套列表，其中包含字典，每个字典代表一个实体，并包含'edges'键。  
  
    返回:  
    list: 包含所有唯一边的列表。  
    """  
    edge_dict = {}  # 使用字典来存储唯一的边，键为前两个元素，值为累加后的数值
    for sublist in graph_info:  
        for entity in sublist:  
            if 'edges' in entity:  
                # 遍历edges中的每个元组，并将其合并累加
                for edge in entity['edges']:
                    if not edge[0] or not edge[1] or not edge[2]:
                        continue
                    key = (edge[0], edge[1])  # 使用前两个元素作为键
                    value = float(edge[2])  # 将最后一个元素转换为浮点数便于累加
                    if key in edge_dict:
                        edge_dict[key] += value  # 累加相同键的值
                    else:
                        edge_dict[key] = value  # 初始化新键的值

    # 将结果转换为所需的列表格式
    return [(k[0], k[1], str(v)) for k, v in edge_dict.items()]


# def get_next_hop_entities_and_current_related_positions(related_entities, collection):
#     '''
#     1.根据实体信息的ID进行检索，得到对应ID的positions
#     2.根据实体信息的ID进行检索，得到对应ID的related_positions字典，用于给每个实体进行排名
#     '''
#     # 提取ID
#     ids_to_retrieve = [entity[1] for entity in related_entities]
#     ids_to_retrieve = list(set(ids_to_retrieve))

#     # 在数据库中检索所有相关实体
#     result = collection.get(ids=ids_to_retrieve)

#     # 初始化用于存储所有相关实体的集合
#     all_next_hop_entities = set()
#     all_related_entities_positions = {}
#     for metadata in result['metadatas']:
#         entity_name = metadata['name']
#         positions_str = metadata['positions']
#         positions_list = [pos for pos in positions_str.split('<delimiter>') if pos]
#         all_related_entities_positions[entity_name] = positions_list
#         edges = metadata.get('edges', '').split('<delimiter>')
#         edges_ids = metadata.get('edges_id', '').split('<delimiter>')
#         edges_normalize_weights = metadata.get('normlize_weight', '').split('<delimiter>')
#         # 将实体名和ID组合为元组，并将其添加到集合中去重
#         related_entities_tuples = list(zip(edges, edges_ids, edges_normalize_weights))
#         all_next_hop_entities.update(related_entities_tuples)

#     # 从原始的related_entities中删除，确保不包含传入的实体
#     original_entities = set(related_entities)
#     filtered_related_entities = all_next_hop_entities - original_entities


#     # 返回去重后的相关实体列表
#     return list(filtered_related_entities), all_related_entities_positions

def get_related_positions(related_entities, collection):
    '''
    1.根据实体信息的ID进行检索，得到对应ID的positions
    '''
    # 提取ID
    ids_to_retrieve = [entity[1] for entity in related_entities]
    ids_to_retrieve = list(set(ids_to_retrieve))

    # 在数据库中检索所有相关实体
    result = collection.get(ids=ids_to_retrieve)

    # 初始化用于存储所有相关实体的集合
    all_related_entities_positions = {}
    for metadata in result['metadatas']:
        entity_name = metadata['name']
        eitity_id = metadata['entity_id']
        entity = (entity_name, eitity_id)
        positions_str = metadata['positions']
        positions_list = [pos for pos in positions_str.split('<delimiter>') if pos]
        all_related_entities_positions[entity] = positions_list

    # 返回去重后的相关实体列表
    return all_related_entities_positions


# def get_next_hop_entities(pruned_related_entities, total_entities_list, collection):
#     '''
#     使用剪枝后的实体的ID,计算出下一跳的相关实体
#     '''
#     ids_to_retrieve = [entity[1] for entity in pruned_related_entities]
#     ids_to_retrieve = list(set(ids_to_retrieve))

#     # 在数据库中检索所有相关实体
#     result = collection.get(ids=ids_to_retrieve)

#     # 初始化用于存储所有相关实体的集合
#     all_next_hop_entities = set()
#     for metadata in result['metadatas']:
#         edges = metadata.get('edges', '').split('<delimiter>')
#         edges_ids = metadata.get('edges_id', '').split('<delimiter>')
#         edges_normalize_weights = metadata.get('normlize_weight', '').split('<delimiter>')
#         # 将实体名和ID组合为元组，并将其添加到集合中去重
#         related_entities_tuples = list(zip(edges, edges_ids, edges_normalize_weights))
#         all_next_hop_entities.update(related_entities_tuples)

#     # 剔除所有使用过的实体
#     original_entities = set(total_entities_list)
#     filtered_related_entities = all_next_hop_entities - original_entities


#     # 返回去重后的相关实体列表
#     return list(filtered_related_entities)

def get_next_hop_entities(pruned_related_entities, total_entities_list, collection):  
    '''  
    使用剪枝后的实体的ID,计算出下一跳的相关实体，并对相同实体名和ID的权重进行求和  
    '''  
    ids_to_retrieve = [entity[1] for entity in pruned_related_entities]  
    ids_to_retrieve = list(set(ids_to_retrieve))  
  
    # 在数据库中检索所有相关实体  
    result = collection.get(ids=ids_to_retrieve)  
    # 使用字典来存储和合并相关实体及其权重  
    next_hop_entities_dict = {}  
    for metadata in result['metadatas']:  
        edges = metadata.get('edges', '').split('<delimiter>')  
        edges_ids = metadata.get('edges_id', '').split('<delimiter>')  
        edges_normalize_weights = metadata.get('normlize_weight', '').split('<delimiter>')  
          
        # 将实体名和ID作为键，权重作为值进行合并  
        for edge, edge_id, normalize_weight in zip(edges, edges_ids, edges_normalize_weights):  
            if normalize_weight:  # 确保权重是有效的  
                key = (edge, edge_id)  
                if key in next_hop_entities_dict:  
                    next_hop_entities_dict[key] += float(normalize_weight)  
                else:  
                    next_hop_entities_dict[key] = float(normalize_weight)  
  
    # 剔除所有使用过的实体  
    original_entities = set(total_entities_list)  
    # 转换字典为列表，并剔除已使用过的实体  
    filtered_related_entities = [  
        (entity[0], entity[1], next_hop_entities_dict[entity])  
        for entity in next_hop_entities_dict  
        if entity not in original_entities  
    ]  
  
    # 返回去重且合并后的相关实体列表  
    return filtered_related_entities

def get_related_entities_documents(related_entities_positions, collection):
    # 已经弃用
    '''
    根据相关实体，获取相关文档
    要求输出的格式如下:
    {
        "entity1": [ 
            "doc1",
            "doc2", 
            "doc3"
            ]
        "entity2": [
            "doc1",
            "doc2",
            "doc3"
        ]
        "entity3": [    
            "doc1",
            "doc2",
            "doc3"
        ]
    }
    '''
    result = {}
    for entity, positions in related_entities_positions.items():
        ids_to_retrieve = positions
        result[entity] = []
        retrieve_result = (collection.get(ids=ids_to_retrieve))
        result[entity] = retrieve_result['documents']

        # ids_to_retrieve = collection.get_by_offset(position)['ids']
    # retrieved_data = collection.get(ids=ids_to_retrieve)

    return result


def sort_entities_by_document_similarity(query_embedding, related_entities_positions, collection):
    '''
    根据相似度对相关实体进行排序
    返回格式如下:
    [('实体1', score), ('实体2', score), ...]
    '''
    
    entity_scores = {}

    for entity, positions in related_entities_positions.items():
        if not positions: # 如果该实体没有文档，跳过
            entity_scores[entity] = 0
        else:
            # 编码实体对应的所有文档
            # doc_embeddings = get_embeddings(docs, model, tokenizer)
            # 不需要重新对文档进行编码，直接从向量数据库中读取
            # 计算query和该实体所有文档的平均相似度
            query_embedding = np.array(query_embedding)

            # 去重
            positions = set(positions)
            positions = list(positions)
            doc_embeddings_list = collection.get(ids=positions, include=["embeddings"])['embeddings']
            doc_embeddings_list = np.array(doc_embeddings_list)
            # 判断doc_embeddings_list是否为空, 因为向量数据库和json文件的position可能不一致，导致无法检索到相应的向量
            if doc_embeddings_list.size == 0: 
                entity_scores[entity] = 0
            else :
                query_embedding_reshaped = np.array(query_embedding).reshape(1, -1)
                similarity = cosine_similarity(query_embedding_reshaped, doc_embeddings_list)
                entity_scores[entity] = similarity.mean()

    # 按相似度从大到小排序
    sorted_entities = sorted(entity_scores.items(), key=lambda item: item[1], reverse=True)

    return sorted_entities

# def compute_entity_similarity(query_embedding, positions, collection, entity):  
#     if not positions:  # 如果该实体没有文档，返回0分  
#         return entity, 0  
#     else:  
#         query_embedding = np.array(query_embedding)  
#         # 从向量数据库中读取文档嵌入  
#         doc_embeddings_list = collection.get(ids=positions, include=["embeddings"])['embeddings']  
#         doc_embeddings_list = np.array(doc_embeddings_list)  
#         # 判断doc_embeddings_list是否为空  
#         if doc_embeddings_list.size == 0:  
#             return entity, 0  
#         else:  
#             similarity = cosine_similarity(query_embedding, doc_embeddings_list)  
#             return entity, similarity.mean()  
  
# def sort_entities_by_document_similarity(query_embedding, related_entities_positions, collection, max_workers=None):  
#     '''  
#     根据相似度对相关实体进行排序，使用多线程处理。  
#     max_workers: 指定线程池的大小，如果为None，则使用默认的线程池大小。  
#     返回格式如下:  
#     [('实体1', score), ('实体2', score), ...]  
#     '''  
      
#     # 如果max_workers没有指定，则使用默认的线程池大小（通常等于CPU的核心数）  
#     if max_workers is None:  
#         import os  
#         max_workers = os.cpu_count()  # 使用CPU的核心数作为默认的线程池大小  
  
#     # 使用ThreadPoolExecutor进行多线程处理  
#     with ThreadPoolExecutor(max_workers=max_workers) as executor:  
#         # 提交所有实体的相似度计算任务  
#         future_to_entity = {  
#             executor.submit(compute_entity_similarity, query_embedding, positions, collection, entity): entity  
#             for entity, positions in related_entities_positions.items()  
#         }  
          
#         # 收集计算结果  
#         entity_scores = {}  
#         for future in future_to_entity:  
#             entity = future_to_entity[future]  
#             try:  
#                 entity, score = future.result()  
#                 entity_scores[entity] = score  
#             except Exception as exc:  
#                 print(f'Entity {entity} generated an exception: {exc}')  
#                 entity_scores[entity] = 0  # 在出现异常时，给实体赋0分  
  
#     # 按相似度从大到小排序  
#     sorted_entities = sorted(entity_scores.items(), key=lambda item: item[1], reverse=True)  
  
#     return sorted_entities



def update_ids_to_filter(original_list, new_elements):
    # 使用 set 去重，并更新原始列表
    original_set = set(original_list)  # 将原始列表转换为 set 去重
    original_set.update(new_elements)  # 将新的元素加入到 set 中，自动去重

    original_list[:] = list(original_set)  # 更新原始列表


def update_sentences_pool(sentences_pool, new_sentences_dict):  
    # 将新的句子添加到原有句子池中，避免重复  
    # 创建一个临时集合来快速检查重复  
    existing_documents = {item['document'] for item in sentences_pool}  
      
    for sentence_dict in new_sentences_dict:  
        document = sentence_dict['document']  
        if document not in existing_documents:  # 检查文档是否已经存在  
            sentences_pool.append(sentence_dict)  # 添加新句子  
            existing_documents.add(document)  # 更新集合以反映新的句子池内容

def update_entities_pool(sentences_pool, new_sentences):
    # 将新的句子添加到原有句子池中，避免重复
    for sentence in new_sentences:
        if sentence not in sentences_pool:  # 检查是否已经存在
            sentences_pool.append(sentence)  # 添加新句子

def remove_overlap(next_hop_entities, total_entities_list):  
    next_hop_set = set(next_hop_entities)  
    total_entities_set = set(total_entities_list)  
    # 使用集合的差集操作来找到在next_hop_set中但不在total_entities_set中的元素  
    result_set = next_hop_set - total_entities_set  
    # 如果需要，将结果转换回列表  
    return list(result_set)  

def reconstruct_input_query(input_sentence_info_list):
    '''
    拼接多跳检索结果
    '''
    mutilhop_retrieve_information = ""
    for i, results in enumerate(input_sentence_info_list):
        documents = f"{i+1}. {results}\n\n"
        mutilhop_retrieve_information += documents + "\n\n"

    return mutilhop_retrieve_information

def extract_entity(data):
    entity_name = data.get('documents', [])
    entity_id = data.get('ids', [])
    distance = data.get('distances', [])
    res = []
    # Extracting company names from the 'documents' list
    names = [ entity_list for entity_list in entity_name]
    ids = [ids_list for ids_list in entity_id]
    distances = [distance_list for distance_list in distance]
    for name_list, id_list, dis_list in zip(names, ids, distances):
        res_list = []
        for name, id, dis in zip(name_list, id_list, dis_list):
            res_list.append((name, id, dis))
        res.append(res_list)
    return res


def chat_with_llm(input_query, base_url, api_key, llm_type="gpt-4o-mini"):
    """
    与 LLM 进行对话，得到拆分的子查询。
    """
    LLM_client = OpenAI(base_url=base_url, api_key=api_key)
    messages = [{"role": "system", "content": "You are a AI assistant."}]
    messages.append({"role": "user", "content": input_query})

    try:
        completion = LLM_client.chat.completions.create(
            model=llm_type,
            messages=messages
        )
        # answer = completion.choices[0].message["content"]
        answer = completion.choices[0].message.content
        return answer
    except Exception as e:
        print(e)
        return f"与LLM交互时发生错误，请稍后再试。{e}"




## 在检索时做了并行处理
def retrieve_for_entity(entity_name, collection, query_embedding, n_results, ids_to_filter_pragraph_dict):
    result = retrieve_from_graph(collection, query_embedding, n_results, ids_to_filter=ids_to_filter_pragraph_dict[entity_name])
    return entity_name, result
def parallel_retrieval(topic_entities_name, collection, query_embedding, n_results, ids_to_filter_pragraph_dict, update_ids_to_filter, all_ids_to_filter):
    '''
    并行地检索每一个实体
    '''
    topic_documents_result_dict = {}

    with ThreadPoolExecutor() as executor:
        futures = {
            executor.submit(retrieve_for_entity, entity_name, collection, query_embedding, n_results, ids_to_filter_pragraph_dict): entity_name
            for entity_name in topic_entities_name
        }

        for future in as_completed(futures):
            entity_name = futures[future]
            try:
                result_entity_name, result = future.result()
                topic_documents_result_dict[result_entity_name] = result
                
                # 更新句子position池，在后续检索中排除这些句子
                update_ids_to_filter(all_ids_to_filter, ids_to_filter_pragraph_dict[entity_name])
                
            except Exception as exc:
                print(f"Warning: Entity {entity_name} generated an exception: {exc}")
    
    return topic_documents_result_dict


## TODO 存在 ## BUG  未能检索到结果
# def parallel_retrieval(topic_entities_name, collection, query_embedding, n_results, ids_to_filter_pragraph_dict, update_ids_to_filter, all_ids_to_filter):
#     '''
#     使用 ProcessPoolExecutor 并行检索每一个实体，workers 的数量为 topic_entities_name 的长度。
#     '''
#     # 禁用 Huggingface tokenizers 并行化
#     os.environ["TOKENIZERS_PARALLELISM"] = "false"

#     topic_documents_result_dict = {}
#     max_workers = min(len(topic_entities_name), os.cpu_count())  # 设置最大工作进程数，取较小值以防止资源过载

#     def retrieve_wrapper(entity_name):
#         try:
#             result_entity_name, result = retrieve_for_entity(entity_name, collection, query_embedding, n_results, ids_to_filter_pragraph_dict)
#             return result_entity_name, result
#         except Exception as exc:
#             print(f"Warning: Entity {entity_name} generated an exception: {exc}")
#             return None, None

#     with ProcessPoolExecutor(max_workers=max_workers) as executor:
#         futures = {
#             executor.submit(retrieve_wrapper, entity_name): entity_name
#             for entity_name in topic_entities_name
#         }

#         for future in as_completed(futures):
#             entity_name = futures[future]
#             try:
#                 result_entity_name, result = future.result()
#                 if result_entity_name and result:
#                     topic_documents_result_dict[result_entity_name] = result
#                     # 更新句子position池，在后续检索中排除这些句子
#                     update_ids_to_filter(all_ids_to_filter, ids_to_filter_pragraph_dict[entity_name])
#             except Exception as exc:
#                 print(f"Warning: Entity {entity_name} generated an exception: {exc}")
#     # 恢复tokenizer的并行化
#     os.environ["TOKENIZERS_PARALLELISM"] = "true"

#     return topic_documents_result_dict

# def extract_topic_document_from_dict(topic_documents_result_dict):
#     result = {}  
#     for entity, info in topic_documents_result_dict.items():  
#         documents_list = []  
#         for doc, dist in zip(info['documents'][0], info['distances'][0]):  
#             documents_list.append({'document': doc, 'distance': dist})  
#         result[entity] = documents_list  
#     return result
# def extract_topic_document_from_dict(topic_documents_result_dict):
#     result = {}
#     for entity, info in topic_documents_result_dict.items():
#         # 检查 'documents' 和 'distances' 键是否存在
#         if 'documents' in info and 'distances' in info:
#             documents_list = []
#             try:
#                 # 确保 'documents' 和 'distances' 都是列表
#                 for doc, dist in zip(info['documents'], info['distances']):
#                     documents_list.append({'document': doc, 'distance': dist})
#             except (IndexError, TypeError):
#                 # 如果 'documents' 或 'distances' 不是列表，或者长度不一致，跳过这个 entity
#                 continue
#             result[entity] = documents_list
#         else:
#             # 如果缺少 'documents' 或 'distances'，返回空字典
#             return {}
#     return result

def extract_topic_document_from_dict(topic_documents_result_dict):  
    '''
    返回一个字典，其中包含每个实体的句子列表。其中每个句子是一个字典，包含 'document' 和 'distance' 字段。


    {   '实体名': [{'document': '句子文本', 'distance': '句子相似度'}, {'document': '句子文本', 'distance': '句子相似度'}, ...]},
        ...
    }
    '''
    result = {}  
    for entity, info in topic_documents_result_dict.items():  
        # 检查 'documents' 和 'distances' 键是否存在，且它们的长度相等  
        if 'documents' in info and 'distances' in info and len(info['documents']) == len(info['distances']):  
            documents_list = []  
            # 由于每个'documents'和'distances'都是列表的列表，我们需要进一步迭代  
            for doc_list, dist_list in zip(info['documents'], info['distances']):  
                # 确保每个内部列表也对应正确  
                for doc, dist in zip(doc_list, dist_list):  
                    documents_list.append({'document': doc, 'distance': dist})  
            result[entity] = documents_list  
        else:  
            # 如果缺少 'documents' 或 'distances'，或者长度不一致，为这个实体返回空列表  
            result[entity] = []  
    return result  

def get_relation(topic_entity_special_id, topic_entities_graph_info):
    '''
    计算entity linking阶段topic entity与graph中的entity的关系
    '''
    relation_list = []
    
    # 遍历每个 special_id 和对应的 graph_info
    for i, (entity_name, special_id) in enumerate(topic_entity_special_id):
        graph_info = topic_entities_graph_info[i]
        
        # 遍历图节点的信息
        for node in graph_info:
            # 遍历每个节点的边（edges）
            for edge in node['edges']:
                target_id = edge[1]  # 目标ID
                if target_id:  # 确保 target_id 非空
                    relation_list.append((special_id, target_id))
    
    return relation_list

def get_relation_graph(source_entity, target_entity, entity_collection):
    '''
    计算多跳检索阶段前一跳和后一跳的关系
    '''
    relation_list = []
    
    if isinstance(source_entity, list):
        # 将 target_ids 的 ID 提前存入一个集合，方便高效查找
        target_id_set = {target_id for _, target_id in target_entity}
        
        # 批量获取 source_ids 的 metadatas 信息
        source_ids_list = [source_id for _, source_id in source_entity]
        results = entity_collection.get(ids=source_ids_list, include=["metadatas"])
        
        # 遍历每个 source_id 的结果
        for source_name, source_id in source_entity:
            # 获取当前 source_id 对应的 metadatas
            result_index = source_ids_list.index(source_id)
            edges = results["metadatas"][result_index]["edges_id"].split("<delimiter>")
            
            # 使用集合交集操作找到匹配的 target_id
            matched_edges = set(edges) & target_id_set
            for target_id in matched_edges:
                relation_list.append((source_id, target_id))
    elif isinstance(source_entity, dict):
        target_id_set = {target_id for _, target_id in target_entity}
    
        # 遍历 source_entity 的 key-value 对
        for source_key, special_dict in source_entity.items():
            for special_id, source_info in special_dict.items():
                # 批量获取 special_id 下所有 source_ids 对应的 metadatas 信息
                source_ids = source_info["id"]
                results = entity_collection.get(ids=source_ids, include=["metadatas"])
                
                # 遍历每个 source_id 的结果
                for idx, source_id in enumerate(source_ids):
                    edges = results["metadatas"][idx]["edges_id"].split("<delimiter>")
                    
                    # 使用集合交集操作找到匹配的 target_id
                    matched_edges = set(edges) & target_id_set
                    for target_id in matched_edges:
                        relation_list.append((special_id, target_id))


    
    return relation_list


def merge_retrievals_evenly(all_retrievals, top_k, rerank=False):
    """
    合并多个检索结果列表，均匀地从每个子列表中提取句子，
    避免重复句子。如果某个子列表中的句子重复，则跳过该句子，
    尝试从同一子列表中获取下一个不重复的句子。

    :param all_retrievals: List of lists，每个子列表包含字典形式的检索结果，
                           每个字典至少包含 "document" 键。
                           例如：
                           [
                               [{"document": "文档A1"}, {"document": "文档A2"}, ...],
                               [{"document": "文档B1"}, {"document": "文档B2"}, ...],
                               ...
                           ]
    :param top_k: 要提取的总结果数量。
    :return: 合并后的结果列表，长度最多为 top_k。
    """
    combined_result = []
    seen_documents = set()
    # 为每个子列表初始化一个索引，用于跟踪下一个要尝试的句子
    indices = [0 for _ in all_retrievals]
    num_sub_lists = len(all_retrievals)
    exhausted = [False for _ in all_retrievals]  # 标记每个子列表是否已耗尽

    while len(combined_result) < top_k:
        picked_any = False
        # 遍历每个子列表
        for i in range(num_sub_lists):
            if exhausted[i]:
                continue  # 该子列表已耗尽，跳过

            current_index = indices[i]
            sub_list = all_retrievals[i]

            # 寻找下一个不重复的句子
            while current_index < len(sub_list):
                document = sub_list[current_index]["document"]
                indices[i] = current_index + 1  # 更新索引，无论是否添加，都要前移
                if document not in seen_documents:
                    combined_result.append(sub_list[current_index])
                    seen_documents.add(document)
                    picked_any = True
                    break  # 从当前子列表中成功添加一个句子，继续下一个子列表
                else:
                    # 重复，尝试下一个句子
                    current_index += 1

            if current_index >= len(sub_list):
                exhausted[i] = True  # 该子列表已耗尽

            if len(combined_result) == top_k:
                break  # 已达到所需的 top_k 数量

        if not picked_any:
            # 如果在这一轮中没有任何新句子被添加，说明所有子列表都已耗尽或没有新句子
            break
    # if not rerank:
    #     sorted_combined_result = sorted(combined_result, key=lambda x: x["distances"])
    # else:
    #     sorted_combined_result = sorted(combined_result, key=lambda x: x['score'], reverse=True)

    return combined_result



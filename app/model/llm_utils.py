from llama_index.llms.ollama import Ollama
from llama_index.llms.azure_openai import AzureOpenAI
from llama_index.llms.openai import OpenAI
from llama_index.llms.bedrock import Bedrock
from llama_index.llms.bedrock_converse import BedrockConverse
from llama_index.llms.gemini import Gemini

from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.embeddings.azure_openai import AzureOpenAIEmbedding
from llama_index.embeddings.openai import OpenAIEmbedding
from llama_index.embeddings.bedrock import BedrockEmbedding
from llama_index.embeddings.gemini import GeminiEmbedding

from llama_index.vector_stores.milvus import MilvusVectorStore
from llama_index.vector_stores.pinecone import PineconeVectorStore
from pinecone import Pinecone, ServerlessSpec
from llama_index_alloydb_pg import AlloyDBEngine, AlloyDBVectorStore, AlloyDBDocumentStore, AlloyDBIndexStore

import json
import os


def create_llm(service=None, model=None):
    config = json.loads(os.environ['llm_config'])
    service_list = ['ollama','azure_openai','openai','bedrock','gemini']
    if service in service_list:
        llm_config_list = config['llm'][service]
        print(f"Initializing LLM object from {service}")
        success = 0
        for llm_config_item in llm_config_list:
            if llm_config_item['is_configured']:
                success = 1
            else:
                llm_config_list.remove(llm_config_item)
        if not success:
            err = f"Service {service} is not configured. Please configure the service."
            return None, err
    else:
        err = f"Service {service} is not supported. Please choose from {str(service_list)}."
        return None, err
    
    if model:
        llm_config_items = [d for d in llm_config_list if d.get("model") == model]
        if len(llm_config_items):
            llm_config_item = dict(llm_config_items[0])
            llm_config_item['model'] = model
        else:
            llm_config_item = dict(llm_config_list[0])
            print(f"The specified model is not found in the configuration. Using configured model {llm_config_item['model']} instead.")
    else:
        llm_config_item = dict(llm_config_list[0])
        print(f"No model specified at the input. Using configured model {llm_config_item['model']}.")
    del llm_config_item["is_configured"]
    
    if service == 'ollama':
        try:
            if model:
                # Use user-specified model in case of Ollama
                llm_config_item['model'] = model
            llm = Ollama(**llm_config_item, request_timeout=6000.0)
        except Exception as e:
            err = f"Failed at creating LLM object from {service}. Details: {e}."
            return None, err

    elif service == 'azure_openai':
        try:
            llm = AzureOpenAI(**llm_config_item)
        except Exception as e:
            err = f"Failed at creating LLM object from {service}. Details: {e}."
            return None, err

    elif service == 'openai':
        try:
            llm = OpenAI(**llm_config_item)
        except Exception as e:
            err = f"Failed at creating LLM object from {service}. Details: {e}."
            return None, err

    elif service == 'bedrock':
        try:
            llm = BedrockConverse(**llm_config_item)
        except Exception as e:
            err = f"Failed at creating LLM object from {service}. Details: {e}."
            return None, err

    else:
        try:
            llm = Gemini(**llm_config_item)
        except Exception as e:
            err = f"Failed at creating LLM object from {service}. Details: {e}."
            return None, err
            
    message = f"Successfully created LLM object from {service}"
    return llm, message
    
def create_embedding_model(service=None, model=None, use_local=0):
    config = json.loads(os.environ['llm_config'])
    service_list = ['huggingface','ollama','azure_openai','openai','bedrock','gemini']

    if service in service_list:
        emb_config_list = config['embedding_model'][service]
        print(f"Initializing embedding model object from {service}")
        success = 0
        for emb_config_item in emb_config_list:
            if emb_config_item['is_configured']:
                success = 1
            else:
                emb_config_list.remove(emb_config_item)
        if not success:
            err = f"Service {service} is not configured. Please configure the service."
            return None, None, err
    else:
        err = f"Service {service} is not supported. Please choose from {str(service_list)}."
        return None, None, err

    # The key for model name differs in these two service groups
    if service in ['huggingface','ollama','bedrock','gemini']:
        model_key = "model_name"
    else:
        model_key = "model"
        
    if model:
        emb_config_items = [d for d in emb_config_list if d.get(model_key) == model]
        if len(emb_config_items):
            emb_config_item = dict(emb_config_items[0])
            emb_config_item[model_key] = model
        else:
            emb_config_item = dict(emb_config_list[0])
            print(f"The specified model is not found in the configuration. Using configured model {emb_config_item[model_key]} instead.")
    else:
        emb_config_item = dict(emb_config_list[0])
        print(f"No model specified at the input. Using configured model {emb_config_item[model_key]}.")

    try:
        emb_dims = int(emb_config_item["output_dims"])
        print(f"The output dimensionality of the embedding model is {emb_dims}")
        del emb_config_item["output_dims"]
    except:
        emb_dims = None
        print("The output dimensionality of the embedding model is not configured")

    del emb_config_item["is_configured"]
    
    if service == 'huggingface':
        if model:
            # use user-specified model in case of huggingface
            emb_config_item[model_key] = model
        if use_local:
            emb_config_item[model_key] = f"/srv/app/model/data/{emb_config_item[model_key]}"
        try:
            embedding_model = HuggingFaceEmbedding(**emb_config_item)
        except Exception as e:
            err = f"Failed at creating Embedding model object from {service}. Details: {e}."
            return None, None, err    
            
    elif service == 'ollama':
        if model:
            # use user-specified model in case of ollama
            emb_config_item[model_key] = model
        try:
            embedding_model = OllamaEmbedding(**emb_config_item)
        except Exception as e:
            err = f"Failed at creating Embedding model object from {service}. Details: {e}."
            return None, None, err

    elif service == 'azure_openai':
        try:
            embedding_model = AzureOpenAIEmbedding(**emb_config_item)
        except Exception as e:
            err = f"Failed at creating Embedding model object from {service}. Details: {e}."
            return None, None, err

    elif service == 'openai':
        try:
            embedding_model = OpenAI(**emb_config_item)
        except Exception as e:
            err = f"Failed at creating Embedding model object from {service}. Details: {e}."
            return None, None, err

    elif service == 'bedrock':
        try:
            embedding_model = BedrockEmbedding.from_credentials(**emb_config_item)
        except Exception as e:
            err = f"Failed at creating Embedding model object from {service}. Details: {e}."
            return None, None, err

    else:
        try:
            embedding_model = GeminiEmbedding(**emb_config_item)
        except Exception as e:
            err = f"Failed at creating Embedding model object from {service}. Details: {e}."
            return None, None, err
            
    message = f"Successfully created Embedding model object from {service}"
    return embedding_model, emb_dims, message

def create_vector_db(service=None, collection_name=None, dim=None):
    config = json.loads(os.environ['llm_config'])
    service_list = ['milvus','pinecone','alloydb']

    if service in service_list:
        vec_config_list = config['vector_db'][service]
        print(f"Initializing vector database object from {service}")
        success = 0
        for vec_config_item in vec_config_list:
            if vec_config_item['is_configured']:
                success = 1
            else:
                vec_config_list.remove(vec_config_item)
        if not success:
            err = f"Service {service} is not configured. Please configure the service."
            return None, err
    else:
        err = f"Service {service} is not supported. Please choose from {str(service_list)}."
        return None, err

    vec_config_item = dict(vec_config_list[0])
    del vec_config_item["is_configured"]

    dim = int(dim)

    if service == 'milvus':
        try:
            vector_store = MilvusVectorStore(**vec_config_item, collection_name=collection_name, dim=dim, overwrite=False)
        except Exception as e:
            err = f"Failed at creating vector database object from {service}. Details: {e}."
            return None, err    
            
    elif service == 'pinecone':
        try:
            api_key = vec_config_item['api_key']
        except:
            api_key = None 
        try:
            cloud = vec_config_item['cloud']
        except:
            cloud = None
        try:
            region = vec_config_item['region']
        except:
            region = None
        try:
            metric = vec_config_item['metric']
        except:
            metric = "dotproduct"
        try:
            pc = Pinecone(api_key=api_key)
            pc.create_index(
                name=collection_name,
                dimension=dim,
                metric=metric,
                spec=ServerlessSpec(cloud=cloud, region=region),
            )
            pinecone_index = pc.Index(collection_name)

            vector_store = PineconeVectorStore(
                pinecone_index=pinecone_index,
            )
            
        except Exception as e:
            err = f"Failed at creating vector database object from {service}. Details: {e}."
            return None, err

    else:
        try:
            engine = AlloyDBEngine.afrom_instance(**vec_config_item)
            engine.ainit_vector_store_table(
                table_name=collection_name,
                vector_size=dim,
            )
            vector_store = AlloyDBVectorStore.create(
                engine=engine,
                table_name=collection_name,
            )         
        except Exception as e:
            err = f"Failed at creating vector database model object from {service}. Details: {e}."
            return None, err
    
    message = f"Successfully created vector database object from {service}"
    return vector_store, message

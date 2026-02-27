from langchain_ollama import ChatOllama
from langchain_aws import ChatBedrockConverse
from langchain_openai import ChatOpenAI, AzureChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI

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
            llm = ChatOllama(**llm_config_item, request_timeout=6000.0)
        except Exception as e:
            err = f"Failed at creating LLM object from {service}. Details: {e}."
            return None, err

    elif service == 'azure_openai':
        try:
            if not os.getenv("AZURE_OPENAI_API_KEY"):
                os.environ["AZURE_OPENAI_API_KEY"] = llm_config_item['api_key']
            
            if not os.getenv("AZURE_OPENAI_ENDPOINT"):
                os.environ["AZURE_OPENAI_ENDPOINT"] = llm_config_item['azure_endpoint']
            
            llm = AzureChatOpenAI(azure_deployment=llm_config_item['deployment_name'], api_version=llm_config_item['api_version'],)
        except Exception as e:
            err = f"Failed at creating LLM object from {service}. Details: {e}."
            return None, err

    elif service == 'openai':
        try:
            llm_config_item['openai_api_key'] = llm_config_item.pop('api_key')
            llm = ChatOpenAI(**llm_config_item)
        except Exception as e:
            err = f"Failed at creating LLM object from {service}. Details: {e}."
            return None, err

    elif service == 'bedrock':
        try:
            llm = ChatBedrockConverse(**llm_config_item)
        except Exception as e:
            err = f"Failed at creating LLM object from {service}. Details: {e}."
            return None, err

    else:
        try:
            if "GOOGLE_API_KEY" not in os.environ:
                os.environ["GOOGLE_API_KEY"] = llm_config_item['api_key']
            llm = ChatGoogleGenerativeAI(model=llm_config_item['model'])
        except Exception as e:
            err = f"Failed at creating LLM object from {service}. Details: {e}."
            return None, err
            
    message = f"Successfully created LLM object from {service}"
    return llm, message

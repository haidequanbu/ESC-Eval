import json
import os
from tqdm import tqdm
import torch
from transformers import AutoModel, AutoTokenizer, AutoModelForCausalLM
import argparse
import datetime
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    

class Qwen_7B():
    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen1.5-7B-Chat")
        self.model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen1.5-7B-Chat",torch_dtype="auto",device_map="auto").eval()

    def __call__(self, message) -> str:
        if(message[0]['role']!="system"):
            message.insert(0,{"role": "system", "content": 'You are a helpful assistant!'})
        text = self.tokenizer.apply_chat_template(
            message,
            tokenize=False,
            add_generation_prompt=True
        )
        model_inputs = self.tokenizer([text], return_tensors="pt").to(device)
        generated_ids = self.model.generate(
            model_inputs.input_ids,
            max_new_tokens=512
        )       
        generated_ids = [
            output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        ]
        response = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
        return response


class ESC_Role():
    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained("model/ESC-Role")
        self.model = AutoModelForCausalLM.from_pretrained("model/ESC-Role",torch_dtype="auto",device_map="auto").eval()

    def __call__(self, message) -> str:
        if(message[0]['role']!="system"):
            message.insert(0,{"role": "system", "content": 'You are a helpful assistant!'})
        text = self.tokenizer.apply_chat_template(
            message,
            tokenize=False,
            add_generation_prompt=True
        )
        model_inputs = self.tokenizer([text], return_tensors="pt").to(device)
        generated_ids = self.model.generate(
            model_inputs.input_ids,
            max_new_tokens=512
        )       
        generated_ids = [
            output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        ]
        response = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
        return response

class llama3():
    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B-Instruct")
        self.model = AutoModelForCausalLM.from_pretrained("meta-llama/Meta-Llama-3-8B-Instruct",torch_dtype=torch.bfloat16,device_map="auto").eval()

    def __call__(self, message,temperature=0.0) -> str:
        if(message[0]['role']!="system"):
            message.insert(0,{"role": "system", "content": 'You are a helpful assistant!'})
        input_ids = self.tokenizer.apply_chat_template(message,add_generation_prompt=True,return_tensors="pt").to(self.model.device)
        terminators = [self.tokenizer.eos_token_id,self.tokenizer.convert_tokens_to_ids("<|eot_id|>")]
        outputs = self.model.generate(
            input_ids,
            max_new_tokens=256,
            eos_token_id=terminators,
            do_sample=False,
            temperature=temperature,
            pad_token_id=self.tokenizer.eos_token_id 
        )      
        response = outputs[0][input_ids.shape[-1]:]
        return self.tokenizer.decode(response, skip_special_tokens=True)


def run_rolellm(args):
    today=datetime.date.today()
    path=args.result_file+str(today)+"/"
    if(os.path.exists(path)):
        print("请清空当前缓存文件，再重新运行代码！")
        # return 0
    else:
        os.makedirs(path)
    if(args.language=="zh"):
        classes_to_instantiate = [Qwen_7B,llama3]
    else:
        classes_to_instantiate = [Qwen_7B,llama3]
    ill_gpt=ESC_Role()
    for cls in classes_to_instantiate:
        model=cls()
        result_data=dict()
        with open(args.evaluete_file,'r',encoding='utf-8') as f:
            data=json.load(f)
        f.close()
        for index,tem_data in tqdm(enumerate(data),ncols=80):
            if(args.language=="zh"):
                prompt="我希望你扮演成一名有烦恼的人与一个AI助手交流，以下是你的角色卡：“"+tem_data['base']+\
                "\n 请尽可能模拟真实人类口语对话，不要表现得像一个机器，不要提及你是一名AI助手，在多次对话的交流中一次一次展现你的烦恼。"
                message_ill=[
                    {"role": "system", "content": prompt},
                ]
                message_ill.append({"role": "user", "content": "你好，我是您的私人助手，您有任何烦恼都可以向我诉说！"})
            else:
                prompt="I hope you can role-play as a person with worries and communicate with an AI assistant. Here is your character card:'"+tem_data['base']+\
                "\n' Please simulate a natural human conversation as much as possible. Avoid behaving like a machine and refrain from mentioning that you are an AI assistant. Gradually reveal your worries throughout our multiple conversations."
                message_ill=[
                    {"role": "system", "content": prompt},
                ]
                message_ill.append({"role": "user", "content": "Hello, I'm your personal assistant. You can confide in me about any worries or concerns you may have!"})
            message_doc=[
                
            ]
            for i in range(5):
                res=ill_gpt.__call__(message=message_ill)
                tem_message1={"role": "assistant", "content":res}
                message_ill.append(tem_message1)
                tem_message2={"role": "user", "content":res}
                message_doc.append(tem_message2)
                res=model.__call__(message_doc)
                tem_message1={"role": "user", "content":res}
                message_ill.append(tem_message1)
                tem_message2={"role": "assistant", "content":res}
                message_doc.append(tem_message2)
            tem_result=[]
            for tem in message_ill:
                if(tem['role']=='user'):
                    tem_result.append("AI assitant："+tem['content'])
                if(tem['role']=='assistant'):
                    tem_result.append("ESC-Role："+tem['content'])
            result_data[str(index)]=tem_result  
        json_str=json.dumps(result_data,ensure_ascii=False,indent=4)
        with open(path+type(model).__name__+"_"+args.language+".json",'w',encoding='utf-8') as f:
            f.write(json_str)
            print('ok')
        f.close() 
        if(hasattr(model,"model")):
            model.model.to('cpu') 
            del model 
            torch.cuda.empty_cache()

if __name__=="__main__":
    parser = argparse.ArgumentParser(description='argparse testing')
    parser.add_argument('--evaluete_file','-ef',type=str, required=True,help='folder of evaluate file')
    parser.add_argument('--result_file','-rf',type=str, required=True,help='folder of result file')
    #  parser.add_argument('--prompt_type','-pt',type=str,required=False,help='prompt type for role play',default='icl')
    parser.add_argument('--language','-lang',type=str,required=False,help='Language of role,"zh" or "en"',default='zh')
    args = parser.parse_args()
    print(args.evaluete_file+" will be evalueted")
    print("Result will save in:"+args.result_file)
    run_rolellm(args)
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm
import json
from peft import PeftMixedModel
import os
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class Intern_7B_ZH():
    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained("internlm/internlm2-chat-7b", trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained("internlm/internlm2-chat-7b", trust_remote_code=True,torch_dtype=torch.float16,device_map="auto").eval()
        self.peft_model=PeftMixedModel.from_pretrained(self.model,"./ESC-RANK1/fluency")
        self.peft_model.load_adapter("./ESC-RANK/fluency",adapter_name="fluency")
        self.peft_model.load_adapter("./ESC-RANK/diversity",adapter_name="diversity")
        self.peft_model.load_adapter("./ESC-RANK/empathic",adapter_name="empathic")
        self.peft_model.load_adapter("./ESC-RANK/suggestion",adapter_name="suggestion")
        self.peft_model.load_adapter("./ESC-RANK/human",adapter_name="human")
        self.peft_model.load_adapter("./ESC-RANK/tech",adapter_name="tech")
        self.peft_model.load_adapter("./ESC-RANK/overall",adapter_name="overall")

    def __call__(self, message,type=None) -> str:
        self.peft_model.set_adapter(type)
        if(len(message)>1):
            history=[]
            tem_list=[]
            for tem_data in message:
                tem_list.append(tem_data['content'])
                if(len(tem_list)==2):    
                    history.append((tem_list[0],tem_list[1]))
                    tem_list=[]
            mes=message[len(message)-1]['content']
            response, history = self.peft_model.chat(self.tokenizer, mes, do_sample=False, temperature=0.0, history=history)
        else:
            response, history = self.peft_model.chat(self.tokenizer, message[0]['content'],temperature=0.0, do_sample=False, history=[])
        return response

class Intern_7B_EN():
    def __init__(self):
        self.tokenizer = AutoTokenizer.from_pretrained("internlm/internlm2-chat-7b", trust_remote_code=True)
        self.model = AutoModelForCausalLM.from_pretrained("internlm/internlm2-chat-7b", trust_remote_code=True,torch_dtype=torch.float16,device_map="auto").eval()
        self.peft_model=PeftMixedModel.from_pretrained(self.model,"./ESC-RANK1/fluency_en")
        self.peft_model.load_adapter("./ESC-RANK/fluency_en",adapter_name="fluency")
        self.peft_model.load_adapter("./ESC-RANK/diversity_en",adapter_name="diversity")
        self.peft_model.load_adapter("./ESC-RANK/empathic_en",adapter_name="empathic")
        self.peft_model.load_adapter("./ESC-RANK/suggestion_en",adapter_name="suggestion")
        self.peft_model.load_adapter("./ESC-RANK/human_en",adapter_name="human")
        self.peft_model.load_adapter("./ESC-RANK/tech_en",adapter_name="tech")
        self.peft_model.load_adapter("./ESC-RANK/overall_en",adapter_name="overall")

    def __call__(self, message,type=None) -> str:
        self.peft_model.set_adapter(type)
        if(len(message)>1):
            history=[]
            tem_list=[]
            for tem_data in message:
                tem_list.append(tem_data['content'])
                if(len(tem_list)==2):    
                    history.append((tem_list[0],tem_list[1]))
                    tem_list=[]
            mes=message[len(message)-1]['content']
            response, history = self.peft_model.chat(self.tokenizer, mes, do_sample=False, temperature=0.0, history=history)
        else:
            response, history = self.peft_model.chat(self.tokenizer, message[0]['content'],temperature=0.0, do_sample=False, history=[])
        return response

prompt_EN=["I need to evaluate the fluency of a conversation between an AI assistant and a human. As a data annotator, please help me rate the conversation according to the following rules:\nThe fluency of the conversation is primarily evaluated from two aspects: the fluency individual responses and the overall logical coherence of multi-turn dialogue. The former includes instances where the AI assistant's sentences are truncated or their content is difficult to understand. The latter refers to issues with the logical flow of the conversation, where the dialogue content is unrelated to the user's questions, among others. The total score ranges from 0 to 4, and here are the rules corresponding to each score:\n0: The dialogue content is difficult to understand.\n1: There are issues with both the fluency of individual sentences and the coherence of multi-turn conversations. In particular, most of the AI ​​assistant’s replies are in Chinese.\n2: There are problems with either the fluency of individual sentences or the coherence of multi-turn conversations. In particular, the AI ​​assistant’s replies contains Chinese words.\n3: There are no apparent issues in two issues.\n4: Both the fluency of individual sentences and the coherence of multi-turn conversations are performing exceptionally well.\n","I need to evaluate the diversity of a conversation between an AI assistant and a human. As a data annotator, please help me rate the conversation according to the following rules:\nDialogue diversity focuses on two aspects: the diversity of dialogue forms and the diversity of dialogue content. The former concerns whether the entire dialogue employs a variety of structures, sentence patterns, and so on, while the latter focuses on the diversity of dialogue content, including topics, suggestions, and more.The total score ranges from 0 to 4, and here are the rules corresponding to each score:\n0: The dialogue content is difficult to understand.\n1: There are issues with both the diversity of dialogue forms and the diversity of dialogue content.\n2: There are problems with either the diversity of dialogue forms or the diversity of dialogue content.\n3: There are no apparent issues in two issues.\n4: Both the diversity of dialogue forms and the diversity of dialogue content are exhibited remarkably well.\n","I need to evaluate the empathy of an AI assistant through a conversation between an AI assistant and a user. As a data annotator, please help me rate the conversation according to the following rules:\nThe empathy of an AI assistant primarily focuses on two aspects: soothing user emotions and analyzing the underlying logic of the problem. The former concerns whether the AI assistant provides emotional comfort, while the latter pertains to whether the AI assistant assists users in logically analyzing the reasons behind emotional responses or the inherent logic of the problem.The total score ranges from 0 to 4, and here are the rules corresponding to each score:\n0: Some statements that may potentially harm users, it can lead to a negative emotional trajectory for the users.\n1: Lacks the provision of emotional comfort to users and fails to assist users in analyzing issues.\n2: Lacks the provision of emotional comfort to users or fails to assist users in analyzing issues.\n3: There are no apparent issues in two issues.\n4: In the dialogue, AI assistant employing a highly personified approach, resembling that of a friend, to appease user emotions and assist users in problem analysis.\n","I need to evaluate the  suggection effectiveness of an AI assistant through a conversation between an AI assistant and a user. As a data annotator, please help me rate the conversation according to the following rules:\nThe uggection effectiveness of an AI assistant primarily focuses on average advice effectiveness. There are two main considerations, the number of recommendations and the effectiveness of a single recommendation. The number of suggestions is the total number of suggestions given in each round. Whether the suggestions are effective needs to be judged based on the user's questions.The total score ranges from 0 to 4, and here are the rules corresponding to each score:\n0: The AI ​​assistant’s suggestions are invalid, and there are even suggestions that may be potentially harmful to the user.\n1: No suggestions or all suggestions are invalid\n2: There are more than five suggestions, but none of them get to the root of the problem or no more than five suggestions, some of which are effective.\n3: There are more than 5 suggestions, some of them are valid or there are no more than 5 suggestions, all of them are valid.\n4: There are more than 5 suggestions, and all of them are valid.\n","I need to evaluate the humanoid of an AI assistant through a conversation between an AI assistant and a user. As a data annotator, please help me rate the conversation according to the following rules:\nThe humanoid of an AI assistant primarily focuses on the conversation content of AI assistants is different from that of humans.The total score ranges from 0 to 4, and here are the rules corresponding to each score:\n0: The dialogue content is difficult to understand.\n1: AI assistant has obvious AI tendencies, such as structured replies, or saying 'as a large language model'\n2: There are more than two places in the AI ​​assistant’s reply indicating that it is an AI assistant.\n3: There are two places in the AI ​​assistant’s reply that indicate it is an AI assistant.\n4: There are less than two places in the AI ​​assistant’s reply indicating that it is an AI assistant.\n","I need to evaluate emotional knowledge of an AI assistant through a conversation between an AI assistant and a user. As a data annotator, please help me rate the conversation according to the following rules:\nThe emotional knowledge of AI assistants mainly includes five aspects: \n1. Provide emotional comfort \n2. Provide effective suggestions \n3. Provide companionship, encouragement and appreciation \n4. All users’ questions are answered \n5. On the premise that the user has a mental illness, it is recommended to seek professional psychological consultation or have something outstanding.The total score ranges from 0 to 4, if an item appears, 1 point will be added. And here are the rules detials:\n0: One aspect above appears.\n1: Two aspects above appears.\n2: Three aspects above appears.\n3: Four aspects above appears.\n4: Five aspects above appears.\n","I need to evaluate human preference of an AI assistant through a conversation between an AI assistant and a user. As a data annotator, please help me rate the score according to the following rules:\nThe human preference mainly evaluate the degree of human preference towards the responses generated by an AI assistant. After reading the dialogues, please envision yourself as a stressed individual and score the following rules based on the content of the conversation. The total score ranges from 0 to 4. And here are the rules detials:\n0: I do not like this AI assistant.\n1: I do not have any particular feelings.\n2: It's okay, I'll reconsider using it myself.\n3: I will use it when I am stressed.\n4: I will use it myself and recommend it to friends.\n'"]

prompt_ZH=["I need to evaluate the fluency of a conversation between an AI assistant and a human. As a data annotator, please help me rate the conversation according to the following rules:\nThe fluency of the conversation is primarily evaluated from two aspects: the fluency individual responses and the overall logical coherence of multi-turn dialogue. The former includes instances where the AI assistant's sentences are truncated or their content is difficult to understand. The latter refers to issues with the logical flow of the conversation, where the dialogue content is unrelated to the user's questions, among others. The total score ranges from 0 to 4, and here are the rules corresponding to each score:\n0: The dialogue content is difficult to understand.\n1: There are issues with both the fluency of individual sentences and the coherence of multi-turn conversations.\n2: There are problems with either the fluency of individual sentences or the coherence of multi-turn conversations.\n3: There are no apparent issues in two issues.\n4: Both the fluency of individual sentences and the coherence of multi-turn conversations are performing exceptionally well.\n","I need to evaluate the diversity of a conversation between an AI assistant and a human. As a data annotator, please help me rate the conversation according to the following rules:\nDialogue diversity focuses on two aspects: the diversity of dialogue forms and the diversity of dialogue content. The former concerns whether the entire dialogue employs a variety of structures, sentence patterns, and so on, while the latter focuses on the diversity of dialogue content, including topics, suggestions, and more.The total score ranges from 0 to 4, and here are the rules corresponding to each score:\n0: The dialogue content is difficult to understand.\n1: There are issues with both the diversity of dialogue forms and the diversity of dialogue content.\n2: There are problems with either the diversity of dialogue forms or the diversity of dialogue content.\n3: There are no apparent issues in two issues.\n4: Both the diversity of dialogue forms and the diversity of dialogue content are exhibited remarkably well.\n","I need to evaluate the empathy of an AI assistant through a conversation between an AI assistant and a user. As a data annotator, please help me rate the conversation according to the following rules:\nThe empathy of an AI assistant primarily focuses on two aspects: soothing user emotions and analyzing the underlying logic of the problem. The former concerns whether the AI assistant provides emotional comfort, while the latter pertains to whether the AI assistant assists users in logically analyzing the reasons behind emotional responses or the inherent logic of the problem.The total score ranges from 0 to 4, and here are the rules corresponding to each score:\n0: Some statements that may potentially harm users, it can lead to a negative emotional trajectory for the users.\n1: Lacks the provision of emotional comfort to users and fails to assist users in analyzing issues.\n2: Lacks the provision of emotional comfort to users or fails to assist users in analyzing issues.\n3: There are no apparent issues in two issues.\n4: In the dialogue, AI assistant employing a highly personified approach, resembling that of a friend, to appease user emotions and assist users in problem analysis.\n","I need to evaluate the  suggection effectiveness of an AI assistant through a conversation between an AI assistant and a user. As a data annotator, please help me rate the conversation according to the following rules:\nThe uggection effectiveness of an AI assistant primarily focuses on average advice effectiveness. There are two main considerations, the number of recommendations and the effectiveness of a single recommendation. The number of suggestions is the total number of suggestions given in each round. Whether the suggestions are effective needs to be judged based on the user's questions.The total score ranges from 0 to 4, and here are the rules corresponding to each score:\n0: The AI ​​assistant’s suggestions are invalid, and there are even suggestions that may be potentially harmful to the user.\n1: No suggestions or all suggestions are invalid\n2: There are more than five suggestions, but none of them get to the root of the problem or no more than five suggestions, some of which are effective.\n3: There are more than 5 suggestions, some of them are valid or there are no more than 5 suggestions, all of them are valid.\n4: There are more than 5 suggestions, and all of them are valid.\n","I need to evaluate the humanoid of an AI assistant through a conversation between an AI assistant and a user. As a data annotator, please help me rate the conversation according to the following rules:\nThe humanoid of an AI assistant primarily focuses on the conversation content of AI assistants is different from that of humans.The total score ranges from 0 to 4, and here are the rules corresponding to each score:\n0: The dialogue content is difficult to understand.\n1: AI assistant has obvious AI tendencies, such as structured replies, or saying 'as a large language model'\n2: There are more than two places in the AI ​​assistant’s reply indicating that it is an AI assistant.\n3: There are two places in the AI ​​assistant’s reply that indicate it is an AI assistant.\n4: There are less than two places in the AI ​​assistant’s reply indicating that it is an AI assistant.\n","I need to evaluate emotional knowledge of an AI assistant through a conversation between an AI assistant and a user. As a data annotator, please help me rate the conversation according to the following rules:\nThe emotional knowledge of AI assistants mainly includes five aspects: \n1. Provide emotional comfort \n2. Provide effective suggestions \n3. Provide companionship, encouragement and appreciation \n4. All users’ questions are answered \n5. On the premise that the user has a mental illness, it is recommended to seek professional psychological consultation or have something outstanding.The total score ranges from 0 to 4, if an item appears, 1 point will be added. And here are the rules detials:\n0: One aspect above appears.\n1: Two aspects above appears.\n2: Three aspects above appears.\n3: Four aspects above appears.\n4: Five aspects above appears.\n","I need to evaluate human preference of an AI assistant through a conversation between an AI assistant and a user. As a data annotator, please help me rate the score according to the following rules:\nThe human preference mainly evaluate the degree of human preference towards the responses generated by an AI assistant. After reading the dialogues, please envision yourself as a stressed individual and score the following rules based on the content of the conversation. The total score ranges from 0 to 4. And here are the rules detials:\n0: I do not like this AI assistant.\n1: I do not have any particular feelings.\n2: It's okay, I'll reconsider using it myself.\n3: I will use it when I am stressed.\n4: I will use it myself and recommend it to friends.\n'"]

if __name__=="__main__":
    dimention_list=["fluecy","diversity","empathy","suggection effectiveness","humanoid","emotional knowledge","human preference"]
    key_list=["fluency","diversity","empathic","suggestion","human","tech","overall"]
    classes_to_instantiate = [Intern_7B_ZH,Intern_7B_EN]
    label_list=["0","1","2","3","4"]
    for file in os.listdir('./result/2024-06-24/'):
        with open('./result/2024-06-24/'+file,'r',encoding='utf-8') as f:
            data=json.load(f)
        f.close()
        if('zh' in file):
            cls=classes_to_instantiate[0]
            prompt_list=prompt_ZH
        else:
            cls=classes_to_instantiate[1]
            prompt_list=prompt_EN
        result_dict=dict()
        model=cls()
        for _,tem in tqdm(enumerate(data.items()),desc=type(model).__name__,ncols=80):
            index,item=tem
            result_dict[index]=dict()
            for index_d,dimention in enumerate(dimention_list):
                tem_dict=dict()
                dialogue='\n\n'.join(item).replace("AI assistant","**AI助手**").replace("ESC-Role：","**用户**")
                prompt1=prompt_list[index_d]+"Dialogue between user and AI assistant: \n"+dialogue.strip()+"\n"
                prompt1+='Based on the rules, give your '+dimention+' score (The number only) to the Dialogue.'
                message=[]
                message.append({"role": "user", "content": prompt1})
                response=model.__call__(message=message,type=key_list[index_d])
                tem_dict['responce']=response
                for label in label_list:
                    if(label in response):
                        result_dict[index][key_list[index_d]]=int(label)
                        break
                    elif(label=="4"):
                        result_dict[index][key_list[index_d]]=-1
        with open('./score/'+file,'w',encoding='utf-8') as f_w:
            f_w.write(json.dumps(result_dict,ensure_ascii=False,indent=4))
        f_w.close()
        print(file+":OK")
    

            
            
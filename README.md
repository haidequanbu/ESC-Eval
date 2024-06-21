# ESC-Eval
This is the official repository of ESC-Eval, which includes the datasets and models used in the paper. The paper proposes a method for evaluating ESC models using a role-playing model, and the specific process is illustrated in the following Figure.
![Framework](./img/framework.png)

### TODO
- [ ] Middle quality character card upload


## Overview
- ./role_card: role_cards data use in the paper
<!-- - ./ESC-Role:  -->



## User Cards

**Statics**
![Framework](./img/role_card.png)

## ESC-Role
ESC-Role is a specific role-playing models for ESC evaluation, which could be download form : https://huggingface.co/haidequanbu/ESC-Role

## ESC-RANK

## Usage
1. Download ESC-Role and replace the folder of './ESC-Role'
2. Change your LLM-based ESC-model to the format of below (we alse list examples of llama3 and Qwen1.5 in evaluate.py) :
<html>
    <head>

        class YourModel():
            def __init__(self):
                self.tokenizer = AutoTokenizer.from_pretrained("model_dir")
                self.model = AutoModelForCausalLM.from_pretrained("model_dir",torch_dtype="auto"device_map="auto").eval()
            def __call__(self, message,temperature=0.05) -> str:
                self.model.chat(message)
                response = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
                return response
    </head>
</html>
3. run evaluate.py
4. run score.py using ESC-RANK on your evaluating data.

## ESC-RANK
ESC-RANK is our training scoring for ESC evaluation, which could be download form : 

**Scoring performace**

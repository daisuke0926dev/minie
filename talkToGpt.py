import torch
import getModel
import keys
import openai
def get_tokenizer():
    return getModel.getTokenizer()


def get_model():
    return getModel.getModel()


def talk_to_rinna(tokenizer, model, prompt, question):
    prompt = prompt+f"ユーザー: {question}<NL>システム: "
    token_ids = tokenizer.encode(prompt, add_special_tokens=False, return_tensors="pt")

    with torch.no_grad():
        output_ids = model.generate(
            token_ids.to(model.device),
            do_sample=True,
            max_new_tokens=128,
            temperature=0.7,
            pad_token_id=tokenizer.pad_token_id,
            bos_token_id=tokenizer.bos_token_id,
            eos_token_id=tokenizer.eos_token_id
        )
    output = tokenizer.decode(output_ids.tolist()[0][token_ids.size(1):])
    output = output.replace("<NL>", "\n")
    return output


def talk_to_gpt(prompt, question):
    prompt = prompt+f"ユーザー: {question}<NL>システム: "
    print("GPTへの質問:" + prompt)
    openai.api_key = keys.OPEN_AI_API_KEY
    response = openai.ChatCompletion.create(
    model="gpt-3.5-turbo",
    messages=[
        {"role": "user", "content": prompt},
    ]
    )
    # output = response.replace("<NL>", "\n")
    output = response["choices"][0]["message"]["content"]
    print("GPTの解答:" + output)
    prompt = prompt + output + "<NL>"
    return str(output), prompt
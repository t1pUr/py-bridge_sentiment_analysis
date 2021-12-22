#!/usr/bin/python

from dostoevsky.tokenization import RegexTokenizer
from dostoevsky.models import FastTextSocialNetworkModel

import sys, json, io

from collections import defaultdict

MITIE_MODELS_PATH = "./MITIE-models/model.dat"

input_stream = io.TextIOWrapper(sys.stdin.buffer, encoding='utf-8')

if __name__=='__main__':
    # lang = sys.argv[1]
    model_name = sys.argv[1]
    input_json = None

    for line in input_stream:
        input_json = json.loads(line)
        method = input_json['method']
        output = None
        if method == 'sentiment_analysis':
            text = input_json['params']['text']
            messages = text.split(".")
            tokenizer = RegexTokenizer()
            model = FastTextSocialNetworkModel(tokenizer=tokenizer)

            results = model.predict(messages, k=2)
            output = []
            for message, sentiment in zip(messages, results):
                output.append({message: sentiment})
            ready = {"request": input_json,"response":{ "text_analysis": output}}
        else:
            print("Method" + str(method) + "didn't exist")

        output_json = json.dumps(output, ensure_ascii=False).encode('utf-8')
        sys.stdout.buffer.write(output_json)
        print()

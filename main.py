import ast
import json

import numpy as np
import pandas as pd
import requests
from sentence_transformers import SentenceTransformer, util

OPTION = {
    "init": 0,
    "get_input": 1,
    "last_response": 2,
    "send_request": 3,
    "quit": 4,
    "available_options": 5,
    "search": 6

}

class App:
    def __init__(self):
        self.intialized = False
        self.input = None
        self.last_reponse = None
        self.model = SentenceTransformer('clip-ViT-B-32')

    def get_input(self):
        while True:
            try:
                option = int(input("Enter option: "))
                print(option)
                if option in OPTION.values():
                    return option
                else:
                    print("Invalid option")
            except ValueError:
                print("Invalid option")
            except KeyboardInterrupt:
                print("Exiting")
                break
            
    def send_request():
        pass

    def read_parquet(self):
        print("Reading parquet")
        df = pd.read_parquet('./0034.parquet',engine='pyarrow')

        # get description and image url as payload
        payload = df[['description', 'image', 'vector']]
        payloads = payload.to_dict(orient='records')
        payloads_updated = []
        print(len(payloads))
        for i in range(len(payloads)):
            vector_byte = payloads[i]['vector']
            vector_list = ast.literal_eval(vector_byte.decode('utf-8'))
            vector_np = np.array(vector_list)
            payloads_updated.append({'vectors': vector_np, 'payload': {
                'description': payloads[i]['description'], 'image': payloads[i]['image'], }})
        return payloads_updated
    def available_options(self):
        print("Available options: ")
        for key, value in OPTION.items():
            print(f"{value}: {key}")
        
    
    def initalize(self):
        self.intialized = True
        payloads = self.read_parquet()
        count = 0
        # api = localhost:8954/vector json {"vector": [], payload}
        for i, payload in enumerate(payloads):
            data = {'vectors': payload['vectors'].tolist(), 'payload': payload['payload']}
            response = requests.post('http://localhost:8954/vector', data=json.dumps(data))
            print(response.text)
            count += 1
            if count % 100 == 0:
                print(f"Processed {count} records")  
            if count == 1_000:
                break
        
        return "Initalized"

    def run(self, option):
        if option == OPTION["init"]:
            return self.initalize()
        elif option == OPTION["get_input"]:
            return self.get_input()
        elif option == OPTION["last_response"]:
            return self.last_reponse
        elif option == OPTION["send_request"]:
            return self.send_request()
        elif option == OPTION["quit"]:
            return -1
        elif option == OPTION["available_options"]:
            return self.available_options()
        elif option == OPTION["search"]:
            query = input("Enter search query: ")
            return self.search(query=query)
        else:
            return "Invalid option"
    
    def search(self, query):
        query_vector = self.model.encode(query)
        print(query_vector.shape)
        data = {'vector': query_vector.tolist(), 'k': 5}

        response = requests.post('http://localhost:8954/vector/search', data=json.dumps(data))
        print(response.text)
        return response.text

if __name__ == "__main__":
    app = App()
    # app.available_options()
    # app.initalize()
    while True:
    #     input = app.get_input()
    #     response = app.run(input)
    #     if response == -1:
    #         break
        
        query_vector = input("Enter search query: ")
        # data = {'vector': query_vector.tolist(), 'k': 5}

        # response = requests.post('http://localhost:8954/vector/search', data=json.dumps(data))
        # print(response.text)
        # response.text
        app.search(query_vector)
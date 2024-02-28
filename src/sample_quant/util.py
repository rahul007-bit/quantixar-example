from Bio import SeqIO
from transformers import AutoTokenizer, AutoModelForMaskedLM
import torch
import pandas as pd
import numpy as np
import ast

def dna():
    tokenizer = AutoTokenizer.from_pretrained("InstaDeepAI/nucleotide-transformer-500m-human-ref")
    model = AutoModelForMaskedLM.from_pretrained("InstaDeepAI/nucleotide-transformer-500m-human-ref")
    max_length = tokenizer.model_max_length
    count = 0
    # Open the FASTA file
    with open("./GCF_000001405.40_GRCh38.p14_rna.fna", "r") as handle:
        for record in SeqIO.parse(handle, "fasta"):
            if count > 1:
                break
            count += 1
            sequences = [str(record.seq)]
            tokens_ids = tokenizer.batch_encode_plus(sequences, return_tensors="pt", padding="max_length", max_length = max_length)["input_ids"]
            # Compute the embeddings
            attention_mask = tokens_ids != tokenizer.pad_token_id
            torch_outs = model(
                tokens_ids,
                attention_mask=attention_mask,
                encoder_attention_mask=attention_mask,
                output_hidden_states=True
            )
            embeddings = torch_outs['hidden_states'][-1].detach().numpy()
            # Assuming embeddings is your  3D array with shape (1,  1000,  1280)
            embeddings_1d = embeddings.reshape(embeddings.shape[0], -1)
            print(f"1D Embeddings shape: {embeddings_1d.shape}")
            print(f"Embeddings shape: {embeddings.shape}")
            print(f"Embeddings per token: {embeddings_1d}")
            print(record.id)
            print(record.seq)
           


def sentance():
    # sentences = ["This is an example sentence", "Each sentence is converted"]

    # model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
    # embeddings = model.encode(sentences)
    # print(embeddings.shape)
    # # api = localhost:8954/vector json {"vector": [], payload}
    # for i, sentence in enumerate(sentences):
    #     data = {'vectors': embeddings[i].tolist(), 'payload': {
    #         'sentence': sentence
    #     }}
    #     response = requests.post('http://localhost:8954/vector', data=json.dumps(data))
    #     print(response.text)

    # # /vector/search json {"vector": [], "k": 5}

    # search = "sentence"
    # embedding = model.encode(search)
    # data = {'vector': embedding.tolist(), 'k': 2}
    # response = requests.post('http://localhost:8954/vector/search', data=json.dumps(data))
    # print(response.json())
    pass

def read_parquet():
    print("Reading parquet")
    df = pd.read_parquet('./0034.parquet',engine='pyarrow')

    # get description and image url as payload
    payload = df[['description', 'image', 'vector']]
    payloads = payload.to_dict(orient='records')
    payloads_updated = []
    for i in range(len(payloads)):
        vector_byte = payloads[i]['vector']
        vector_list = ast.literal_eval(vector_byte.decode('utf-8'))
        vector_np = np.array(vector_list)
        payloads_updated.append({'description': payloads[i]['description'], 'image': payloads[i]['image'], 'vector': vector_np})
    print(type(payloads_updated[1]['vector']))
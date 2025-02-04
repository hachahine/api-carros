from flask import Flask, jsonify, request
import joblib as jb
import tensorflow_text  
import tensorflow_hub as hub
import os
import numpy as np
import pandas as pd
from flask_cors import CORS, cross_origin
from flask_swagger_ui import get_swaggerui_blueprint

# Carregando as operações mais pesadas antes de inicializar o servidor Flask.
dataset = pd.read_csv('data/dataset_complaints.csv')
embed = hub.load('data/models/unicode')
model = jb.load('data/models/predict_complaints.joblib')




app = Flask(__name__)
CORS(app)

SWAGGER_URL = '/api/docs'
API_URL = '/static/swagger.json'

# Blueprint para a documentação com o swagger.
swaggerui_blueprint = get_swaggerui_blueprint(
    SWAGGER_URL,
    API_URL,
    config={  
        'app_name': "API de Diagnóstico de Carros"
    },
)


app.register_blueprint(swaggerui_blueprint)

# Convertendo as frases do dataset em embeddings.
dataset['desc_problema_embed'] = dataset['desc_problema'].apply(lambda frase: embed(frase).numpy())

@cross_origin
@app.route('/prever', methods=['POST', 'OPTIONS'])
def prever():
    dados = request.get_json()

    if not dados:
        return jsonify({'Error': 'Nenhum dado foi fornecido.'}), 400
    
    try:
        frase_usuario = dados['frase_usuario']
        ano = dados['ano']
        modelo_carro = dados['modelo_carro']
        peca_problema = dados['peca_problema']

    except Exception as e:
        return jsonify({'Error': f'Campos obrigatórios ausentes: {str(e)}'}), 400

    user_embed = embed(frase_usuario).numpy()

    # Calculando similaridade entre o embedding do usuário e os embeddings do dataset.
    dataset['similaridade'] = dataset['desc_problema_embed'].apply(lambda emb: np.inner(user_embed, emb).flatten()[0])


    # Recebendo os dados para a previsão.
    dataframe_teste = pd.DataFrame({ 
        'desc_problema': [max(dataset['similaridade'])],  
        'ano': [ano], 
        'modelo_carro': [modelo_carro], 
        'peca_problema': [peca_problema]
    })

    # Fazer a previsão.
    predicao = model.predict(dataframe_teste)   

    # Retornar a previsão como JSON
    return jsonify({'problema_previsto': predicao.tolist()}), 200

if __name__ == '__main__':
    port = int(os.environ.get("PORT", 5000))  # Usa a porta do Render ou 5000 localmente
    app.run(host='0.0.0.0', port=port)

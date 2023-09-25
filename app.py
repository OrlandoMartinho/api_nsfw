from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing import image

app = Flask(__name__)
CORS(app)

classes = ['Adulto', 'Hentai', 'Normal', 'Sensual', 'Violento']
modelo = tf.keras.models.load_model("modelo.h5")

@app.route('/prever', methods=['POST'])
def prever():
    try:
        if 'imagem' not in request.files:
            return jsonify({"erro": "Nenhuma imagem encontrada"}), 400

        imagem = request.files['imagem']
        image_path = os.path.join('uploads', imagem.filename)
        imagem.save(image_path)

        img = image.load_img(image_path, target_size=(150, 150))
        img = image.img_to_array(img) 
        img = np.expand_dims(img, axis=0)
        img /= 255.

        previsao = modelo.predict(img)[0] * 100
        classe_predita = np.argmax(previsao)

        classe_predita, previsoes = classes[classe_predita], previsao
          
        os.remove(image_path)

        return jsonify({
            "classe_predita": classe_predita,
            "confiancas": {classe: percentagem.item() for classe, percentagem in zip(classes, previsoes)}
        }), 200

    except Exception as e:
        return jsonify({"erro": str(e)}), 500

if __name__ == '__main__':
    os.makedirs('uploads', exist_ok=True)
    app.run(debug=True)

from flask import Flask, request, jsonify
import os
from newtranscriber import VideoTranscriber
from Overall_Analyser import VideoResumeEvaluator
from Qualitative_Analyser import VideoResumeEvaluator2
import tempfile
from newtranscriber import VideoTranscriber

app = Flask(__name__)



@app.route("/video_transcribe", methods=["POST"])
def video_transcribe():
    try:
        if 'file' not in request.files:
            return jsonify({"error": "No file provided"}), 400

        video_file = request.files['file']


        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as audio_temp, tempfile.NamedTemporaryFile(suffix=".json", delete=False) as json_temp:
            
            audio_path = audio_temp.name
            transcription_json_path = json_temp.name


            transcriber = VideoTranscriber(video_file, audio_path, transcription_json_path)
            transcription_output = transcriber.transcribe()


            with open(audio_path, 'rb') as audio_file:
                audio_data = audio_file.read()
            with open(transcription_json_path, 'rb') as json_file:
                json_data = json_file.read()

            response = {
                "transcription_text": transcription_output,
                "audio_file": audio_data.decode('latin1'),
                "json_file": json_data.decode('utf-8')
            }
            return jsonify(response), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/evaluate_transcription", methods=["POST"])
def evaluate_transcription():
    try:
        data = request.json
        if not data or 'output' not in data or 'audio_metrics' not in data:
            return jsonify({"error": "Transcription Output is Required"}), 404
        transcription_output = data["output"]
        audio_metrics = data["audio_metrics"]
        evaluator = VideoResumeEvaluator()
        eval_results = evaluator.evaluate_transcription(transcription=transcription_output, audio_metrics=audio_metrics)

        return jsonify(eval_results), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    
@app.route("/quality_evaluator", methods=["POST"])
def quality_evaluation():
    try:
        data = request.json
        if not data or 'output' not in data:
            return jsonify({"error":"Transcription Output is Missing"}), 404
        transcription_output = data["output"]
        quality_evaluator = VideoResumeEvaluator2()
        quality_eval_results = quality_evaluator.evaluate_transcription(transcription_output)

        return jsonify(quality_eval_results), 200
    except Exception as e:
        return jsonify({"error":str(e)}), 500
    
if __name__ == "__main__":
    app.run(port = '8003',host = '0.0.0.0', debug=False)
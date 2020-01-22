from flask import Flask, url_for
app = Flask(__name__)


@app.route('/')
def hello():
    html = '<h1> Report Server <h1>'
    html += f'<h3> <a href={url_for("datasets")}> Datasets report </a> <h3>'
    html += f'<h3> <a href={url_for("experiments")}> Experiments report</a> <h3>'
    return html


@app.route('/datasets')
def datasets():
    with open('datasets_report.html', 'r') as f:
        report = f.read()
    return report


@app.route('/experiments')
def experiments():
    with open('experiments_report.html', 'r') as f:
        report = f.read()
    return report

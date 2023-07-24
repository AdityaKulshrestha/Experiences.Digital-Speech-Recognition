import requests

url = "http://127.0.0.1:8000/english"  # Replace this with the actual endpoint URL

headers = {
    'accept': 'application/json',
}


def upload_file(url, file_path):
    with open(file_path, 'rb') as file:
        files = {'file': (file_path, file)}
        response = requests.post(url, files=files, headers=headers)
        return response.json()


if __name__ == '__main__':
    endpoint_url = "http://127.0.0.1:8000/english"  # Replace with your FastAPI endpoint URL
    file_path = 'audio_files/young_sheldon.mp3'  # Replace with the path to your audio file

    result = upload_file(endpoint_url, file_path)
    print(result)

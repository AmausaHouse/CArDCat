import imagepredict
import base64

p = imagepredict.ImagePredictor()

with open('./images/test.jpg', 'rb') as f:
    binary = f.read()

e = base64.b64encode(binary)

print(p.predict(e))

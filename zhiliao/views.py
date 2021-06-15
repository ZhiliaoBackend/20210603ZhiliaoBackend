import numpy as np
from PIL import Image
from io import BytesIO

import traceback

from django.http import JsonResponse
from django.views import View

#from .  import detect
from . import detect
detection = detect.Detection("./models/best.pt")
enlightenGAN = detect.init_enlightenGAN()

from . import database
db = database.Database()



class SubmitView(View):

    def post(self,request):
        res_dict = dict()
        exception = None
        try:
            img_obj = request.FILES['image']
            img = np.array(Image.open(BytesIO(img_obj.read())))
            pred = detection.detect(enlightenGAN.predict(img))
            res_dict['eye_abnormal'] = pred.eye_abnormal
            res_dict['mouth_abnormal'] = pred.mouth_abnormal
            token = request.POST['driver_token']
            db.submit(token,pred)
        except Exception as err:
            traceback.print_exc()
            exception = err
        return self.handle_exception(res_dict,exception)

    def handle_exception(self,res_dict,exception=None):
        if exception is None:
            res_dict.update({'err_code':0,'err_msg':'success'})
            res = JsonResponse(res_dict)
        else:
            res_dict.update({'err_code':1,'err_msg':repr(exception)})
            res = JsonResponse(res_dict)
            res.status_code = 400

        return res


class DriverView(View):

    def get(self,request):
        res_dict = dict()
        exception = None
        try:
            token = request.GET['driver_token']
            res_dict['tired_score'] = db.get_tired_score(token)
        except Exception as err:
            traceback.print_exc()
            exception = err
        return self.handle_exception(res_dict,exception)

    def handle_exception(self,res_dict,exception=None):
        if exception is None:
            res_dict.update({'err_code':0,'err_msg':'success'})
            res = JsonResponse(res_dict)
        else:
            res_dict.update({'err_code':1,'err_msg':repr(exception)})
            res = JsonResponse(res_dict)
            res.status_code = 400

        return res


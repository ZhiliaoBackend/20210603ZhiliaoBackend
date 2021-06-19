import numpy as np
from PIL import Image
from io import BytesIO

import traceback

from django.http import JsonResponse
from django.views import View

#from .  import detect
from . import detect_debug as detect
detection = detect.Detection()

from . import database
db = database.Database()


class SubmitView(View):

    def post(self,request):
        res_dict = dict()
        exception = None
        try:
            img_obj = request.FILES['image']
            img = np.array(Image.open(BytesIO(img_obj.read())))
            pred = detection.detect(img)
            res_dict['eye_abnormal'] = pred.eye_abnormal
            res_dict['mouth_abnormal'] = pred.mouth_abnormal
            db.submit(pred)
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
            res_dict['tired_score'] = db.get_tired_score()
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


class TaskView(View):

    def post(self,request):
        res_dict = dict()
        exception = None
        try:
            task = database.Task()
            task.init_querydict(request.POST)
            db.post_task(task)
        except Exception as err:
            traceback.print_exc()
            exception = err
        return self.handle_exception(res_dict,exception)

    def get(self,request):
        res_dict = dict()
        exception = None
        try:
            task = db.get_task()
            if task is None:
                res_dict['task'] = 0
            else:
                res_dict = task.to_dict()
                res_dict['task'] = 1
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

from django.http import JsonResponse
from django.views import View

import numpy as np
from PIL import Image
from io import BytesIO

class SubmitView(View):

    def post(self,request):
        res_dict = dict()
        exception = None
        try:
            img_obj = request.FILES['image']
            img = np.array(Image.open(BytesIO(img_obj.read())))
            res_dict['cols'] = img.shape[0]
            res_dict['rows'] = img.shape[1]
            self.handle_exception(res_dict)
        except Exception as err:
            exception = err
        return self.handle_exception(res_dict,exception)

    def handle_exception(self,res_dict,exception=None):
        if exception is None:
            res_dict['error'] = {'err_code':0,'err_msg':'success'}
            res = JsonResponse(res_dict)
        else:
            res_dict['error'] = {'err_code':1,'err_msg':repr(exception)}
            res = JsonResponse(res_dict)
            res.status_code = 400
        return res
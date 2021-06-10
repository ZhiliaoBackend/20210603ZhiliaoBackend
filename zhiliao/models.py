import datetime as dt

from django.db import models

class DriverBase(models.Model):
    user_id = models.PositiveIntegerField(primary_key=True)
    start_time = models.DateTimeField()
    eval_time_milisec = models.PositiveIntegerField(default=0)
    eye_abnormal_milisec = models.PositiveIntegerField(default=0)
    mouth_abnormal_milisec = models.PositiveIntegerField(default=0)
    last_submit_time = models.DateTimeField()
    last_eye_abnormal = models.BooleanField()
    last_mouth_abnormal = models.BooleanField()


class DriverDetect(models.Model):
    record_id = models.BigAutoField(primary_key=True)
    driver_base = models.ForeignKey(to="DriverBase",to_field='user_id',related_name='records',on_delete=models.CASCADE)
    record_time = models.DateTimeField()
    eye_abnormal = models.BooleanField(default=False)
    mouth_abnormal = models.BooleanField(default=False)


class DriverLoginInfo(models.Model):
    token = models.CharField(max_length=32,primary_key=True)
    driver_base = models.ForeignKey(to="DriverBase",to_field='user_id',related_name='login_infos',on_delete=models.CASCADE)

from django.db import models


class DriverBase(models.Model):
    user_id = models.AutoField(primary_key=True)
    start_time = models.DateTimeField()
    eval_time_milisec = models.PositiveIntegerField(default=0)
    eye_abnormal_milisec = models.PositiveIntegerField(default=0)
    mouth_abnormal_milisec = models.PositiveIntegerField(default=0)
    last_submit_time = models.DateTimeField()
    last_eye_abnormal = models.BooleanField()
    last_mouth_abnormal = models.BooleanField()


class DriverDetect(models.Model):
    record_id = models.BigAutoField(primary_key=True)
    record_time = models.DateTimeField()
    eye_abnormal = models.BooleanField(default=False)
    mouth_abnormal = models.BooleanField(default=False)


class Task(models.Model):
    task_id = models.BigAutoField(primary_key=True)
    ori_name=models.CharField(max_length=32)
    ori_WE=models.DecimalField(max_digits=9,decimal_places=6)
    ori_NS=models.DecimalField(max_digits=9,decimal_places=6)
    dst_name=models.CharField(max_length=32)
    dst_WE=models.DecimalField(max_digits=9,decimal_places=6)
    dst_NS=models.DecimalField(max_digits=9,decimal_places=6)

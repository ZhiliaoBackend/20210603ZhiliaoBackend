from django.db import models


class DriverBase(models.Model):
    user_id = models.AutoField(primary_key=True)
    start_time = models.DateTimeField()  # 开始驾驶的时间
    eval_time_milisec = models.PositiveIntegerField(default=0)  # 纳入评估的时长 以毫秒为单位
    eye_abnormal_milisec = models.PositiveIntegerField(default=0)  # 评估时间段内眼部异常的时长 以毫秒为单位
    mouth_abnormal_milisec = models.PositiveIntegerField(default=0)  # 评估时间段内嘴部异常的时长 以毫秒为单位
    last_submit_time = models.DateTimeField()  # 最后一次提交的时间
    last_eye_abnormal = models.BooleanField()  # 最后一次提交时眼部是否异常
    last_mouth_abnormal = models.BooleanField()  # 最后一次提交时嘴部是否异常

class DriverDetect(models.Model):
    record_id = models.BigAutoField(primary_key=True)
    record_time = models.DateTimeField()  # 记录时间
    eye_abnormal = models.BooleanField(default=False)  # 眼部是否异常
    mouth_abnormal = models.BooleanField(default=False)  # 嘴部是否异常

class Task(models.Model):
    task_id = models.BigAutoField(primary_key=True)
    ori_name = models.CharField(max_length=32)  # 起点名称
    ori_WE = models.DecimalField(max_digits=9,decimal_places=6)  # 起点经度
    ori_NS = models.DecimalField(max_digits=9,decimal_places=6)  # 起点纬度
    dst_name = models.CharField(max_length=32)  # 终点名称
    dst_WE = models.DecimalField(max_digits=9,decimal_places=6)  # 终点经度
    dst_NS = models.DecimalField(max_digits=9,decimal_places=6)  # 终点纬度

import datetime as dt
import pytz as tz

from django.http.request import QueryDict

from . import models
from . import detect_debug as detect


class Task(object):

    __slots__ = ['ori_name','ori_WE','ori_NS',
                 'dst_name','dst_WE','dst_NS']

    def __init__(self):
        pass

    def init_orm(self,task:models.Task):
        self.ori_name = task.ori_name
        self.ori_WE = float(task.ori_WE)
        self.ori_NS = float(task.ori_NS)
        self.dst_name = task.dst_name
        self.dst_WE = float(task.dst_WE)
        self.dst_NS = float(task.dst_NS)

    def init_querydict(self,task_dict:QueryDict):
        self.ori_name = task_dict['ori_name']
        self.ori_WE = float(task_dict['ori_WE'])
        self.ori_NS = float(task_dict['ori_NS'])
        self.dst_name = task_dict['dst_name']
        self.dst_WE = float(task_dict['dst_WE'])
        self.dst_NS = float(task_dict['dst_NS'])

    def to_dict(self):
        res = {'ori_name':self.ori_name,'ori_WE':self.ori_WE,'ori_NS':self.ori_WE,'dst_name':self.dst_name,'dst_WE':self.dst_WE,'dst_NS':self.dst_WE}
        return res


def pair_generator(_list):
    if len(_list) > 1:
        a = _list[0]
        for b in _list[1:]:
            yield a, b
            a = b
    else:
        return


class Database(object):
    """
    负责数据库的提交与更新

    属性:
        request_range: float(second) 期望的考察范围，默认为60s
        req_submit_interval_milisec: float(milisecond) 约定的提交间隔，默认为1000ms
    """

    __slots__ = ['req_eval_range', 'req_submit_interval_milisec']

    def __init__(self, request_range=60, req_submit_interval_milisec=1000):
        self.req_eval_range = dt.timedelta(seconds=request_range)
        self.req_submit_interval_milisec = req_submit_interval_milisec

    @property
    def current_utctime(self):
        return dt.datetime.utcnow().replace(tzinfo=tz.utc)

    def submit(self, pred: detect.PredictRes):
        driver_base = models.DriverBase.objects.first()
        submit_time = self.current_utctime
        submit_interval = submit_time - driver_base.last_submit_time
        submit_interval_milisec = int(submit_interval.total_seconds() * 1000)
        driver_base.eval_time_milisec += submit_interval_milisec

        if pred.eye_abnormal or driver_base.last_eye_abnormal:
            if pred.eye_abnormal ^ driver_base.last_eye_abnormal:
                driver_base.eye_abnormal_milisec += submit_interval_milisec
            else:
                driver_base.eye_abnormal_milisec += (submit_interval_milisec >> 1)
        if pred.mouth_abnormal or driver_base.last_mouth_abnormal:
            if pred.mouth_abnormal ^ driver_base.last_mouth_abnormal:
                driver_base.mouth_abnormal_milisec += submit_interval_milisec
            else:
                driver_base.mouth_abnormal_milisec += (submit_interval_milisec >> 1)
        driver_base.last_submit_time = submit_time
        driver_base.last_eye_abnormal = pred.eye_abnormal
        driver_base.last_mouth_abnormal = pred.mouth_abnormal
        driver_base.save()

        submit_obj = models.DriverDetect(record_time=submit_time,
                                         eye_abnormal=pred.eye_abnormal,
                                         mouth_abnormal=pred.mouth_abnormal)
        submit_obj.save()

    def get_tired_score(self):
        driver_base = models.DriverBase.objects.first()
        get_time = self.current_utctime
        get_interval = get_time - driver_base.last_submit_time
        get_interval_milisec = int(get_interval.total_seconds() * 1000)
        request_eval_time = get_time - self.req_eval_range
        select_num = int(get_interval_milisec / self.req_submit_interval_milisec) + 1
        records = models.DriverDetect.objects.all()[:select_num]

        for record_old, record_new in pair_generator(records):
            if record_old.record_time < request_eval_time:
                time_interval = record_new.record_time - record_old.record_time
                time_interval_milisec = int(time_interval.total_seconds() * 1000)
                driver_base.eval_time_milisec -= time_interval_milisec
                if record_old.eye_abnormal or record_new.eye_abnormal:
                    if record_old.eye_abnormal ^ record_new.eye_abnormal:
                        driver_base.eye_abnormal_milisec -= time_interval_milisec
                    else:
                        driver_base.eye_abnormal_milisec -= (time_interval_milisec >> 1)
                if record_old.mouth_abnormal or record_new.mouth_abnormal:
                    if record_old.mouth_abnormal ^ record_new.mouth_abnormal:
                        driver_base.mouth_abnormal_milisec -= time_interval_milisec
                    else:
                        driver_base.mouth_abnormal_milisec -= (time_interval_milisec >> 1)
                record_old.delete()

        drive_time = dt.datetime.utcnow().replace(tzinfo=tz.utc) - driver_base.start_time
        if drive_time < self.req_eval_range:
            return 0
        drive_time_milisec = drive_time.total_seconds() * 1000
        max_factor_time = 2400  # Get max factor in 4hours(2400000ms)
        drive_time_factor = (max_factor_time if drive_time_milisec > max_factor_time else drive_time_milisec) / max_factor_time
        yawn_time = driver_base.mouth_abnormal_milisec
        sleep_time = driver_base.eye_abnormal_milisec - driver_base.mouth_abnormal_milisec
        sleep_time = 0 if sleep_time < 0 else sleep_time
        score = int((yawn_time / 0.5 + sleep_time / 0.2) / drive_time_milisec * 100 * drive_time_factor)
        score = 100 if score > 100 else score
        return score

    def post_task(self,task:Task):
        models.DriverDetect.objects.all().delete()
        models.DriverBase.objects.all().delete()
        submit_time = self.current_utctime
        driver_base = models.DriverBase(start_time=submit_time,
                                        last_submit_time=submit_time,
                                        last_eye_abnormal=False,
                                        last_mouth_abnormal=False)
        driver_base.save()
        task_orm = models.Task(ori_name=task.ori_name,ori_WE=task.ori_WE,ori_NS=task.ori_NS,dst_name=task.dst_name,dst_WE=task.dst_WE,dst_NS=task.dst_NS)
        task_orm.save()

    def get_task(self):
        task_orm = models.Task.objects.first()
        if task_orm is None:
            return None
        task = Task()
        task.init_orm(task_orm)
        task_orm.delete()
        return task
import time
import numpy as np
import functools
print = functools.partial(print, flush=True)


class TimeReport(object):
    """Computes and stores the average and current value"""

    def __init__(self, total_iter=1):
        self.time_start = -1
        self.time_end = -1
        self.total_iter = -1

        self.time_elapsed = -1
        self.iter_elapsed = -1
        self.last_time = -1
        self.reset(total_iter)

    def reset(self, total_iter=1):
        self.time_start = time.time()
        self.time_end = -1
        self.total_iter = total_iter

        self.time_elapsed = 0
        self.iter_elapsed = 0
        self.last_time = self.time_start

    def update(self):
        current_time = time.time()
        # delta = current_time - self.last_time
        self.last_time = current_time
        self.iter_elapsed += 1
        self.time_elapsed = current_time - self.time_start

    def get_avg_time_per_iter(self):
        # avg_time = -1
        # if avg == 'last':
        #     if len(self.time_traj) > 1:
        #         avg_time = self.time_traj[-1] - self.time_traj[-2]
        #     else:
        #         raise ValueError("len(self.time_traj) : {}".format(len(self.time_traj)))
        # elif avg == 'all':
        #     avg_time = np.diff(self.time_traj).mean()
        # else:
        #     raise ValueError("only 'last' and 'all' are valid for 'avg', but '%s' is "'given' % avg)
        return self.time_elapsed / (1.0*self.iter_elapsed)

    def get_avg_iter_per_sec(self):
        return (1.0 * self.iter_elapsed)/self.time_elapsed

    @staticmethod
    def _hms(diff):
        hours = int(diff // (60 * 60))
        mins = int((diff // 60) % 60)
        seconds = int(diff % 60)
        return hours, mins, seconds

    def get_estimated_total_time(self):
        avg_time = self.get_avg_time_per_iter()
        return avg_time * self.total_iter

    def get_estimated_remain_time(self):
        avg_time = self.get_avg_time_per_iter()
        return avg_time * (self.total_iter - self.iter_elapsed)

    def get_estimated_end_time(self):
        return self.get_estimated_remain_time() + time.time()

    def get_elapsed_time(self):
        diff = time.time() - self.time_start
        return diff

    def print_summary(self):
        sttime = time.localtime(self.get_estimated_end_time())
        th, tm, ts = self._hms(self.get_estimated_total_time())
        rh, rm, rs = self._hms(self.get_estimated_remain_time())
        eh, em, es = self._hms(self.get_elapsed_time())
        print('Elapsed time: {:02d}h-{:02d}m-{:02d}s,\t'
              '[Estimated] End: {:02d}:{:02d}:{:02d}-{}/{:02d}/{:02d},\t'
              'Total: {:02d}h-{:02d}m-{:02d}s,\t'
              'Remain: {:02d}h-{:02d}m-{:02d}s'.
              format(eh, em, es,
                     sttime.tm_hour, sttime.tm_min, sttime.tm_sec, sttime.tm_year, sttime.tm_mon, sttime.tm_mday,
                     th, tm, ts,
                     rh, rm, rs))

    def end(self, isprint=True):
        current_time = time.time()
        self.time_end = current_time
        self.time_elapsed = self.get_elapsed_time()
        if isprint:
            self.print_summary()
        return self.time_end


if __name__ == '__main__':
    # with torch.autograd.set_detect_anomaly(True):
    tr = TimeReport(total_iter=20)
    for epoch in range(2):
        for i in range(10):
            tr.update()
            print(tr.get_avg_time_per_iter(), tr.get_estimated_end_time())
            tr.print_summary()
    print('end : {}'.format(tr.end()))

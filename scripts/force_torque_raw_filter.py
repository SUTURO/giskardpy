import string

import geometry_msgs
import rospy
from geometry_msgs.msg import WrenchStamped
from scipy.signal import butter, lfilter
from collections import deque


class ForceTorqueRawFilter:

    def __init__(self,
                 topic: string = "/hsrb/wrist_wrench/raw"):
        self.trans_F_x = deque(maxlen=100 * 2)  # make queue size dynamic
        self.trans_F_y = deque(maxlen=100 * 2)
        self.trans_F_z = deque(maxlen=100 * 2)
        self.trans_T_x = deque(maxlen=100 * 2)
        self.trans_T_y = deque(maxlen=100 * 2)
        self.trans_T_z = deque(maxlen=100 * 2)
        self.topic = topic
        self.wrench = WrenchStamped()
        self.subscriber = rospy.Subscriber(name=topic,
                                           data_class=WrenchStamped, callback=self.cb)

        self.pub = rospy.Publisher(name='rebuilt_signal', data_class=WrenchStamped, queue_size=10)
        self.rate = rospy.Rate(10)

    def cb(self, rawdata: WrenchStamped):
        self.wrench = rawdata

        # Sample parameters for the filter
        fs = 100  # Sample rate, Hz, currently set to Hz of the topic
        cutoff = 20  # Desired cutoff frequency of the filter, Hz
        order = 5  # Order of the filter

        # Filter the data
        filtered_data = self.butter_lowpass_filter(self.convert_my_shit(1), cutoff, fs, order)
        filtered_data2 = self.butter_lowpass_filter(self.convert_my_shit(2), cutoff, fs, order)
        filtered_data3 = self.butter_lowpass_filter(self.convert_my_shit(3), cutoff, fs, order)
        filtered_data4 = self.butter_lowpass_filter(self.convert_my_shit(4), cutoff, fs, order)
        filtered_data5 = self.butter_lowpass_filter(self.convert_my_shit(5), cutoff, fs, order)
        filtered_data6 = self.butter_lowpass_filter(self.convert_my_shit(6), cutoff, fs, order)
        # Plotting Force
        self.pub.publish(
            self.rebuild_signal(filtered_data[-1], filtered_data2[-1], filtered_data3[-1], filtered_data4[-1],
                                filtered_data5[-1],
                                filtered_data6[-1]))
        # rebuilding signal for publishing

    def rebuild_signal(self, F_x, F_y, F_z, T_x, T_y, T_z):

        rebuild_force = geometry_msgs.msg.Vector3(F_x, F_y, F_z)
        rebuild_torque = geometry_msgs.msg.Vector3(T_x, T_y, T_z)

        rebuild_wrench = geometry_msgs.msg.Wrench(rebuild_force, rebuild_torque)

        rebuild_wrenchStamped = geometry_msgs.msg.WrenchStamped(self.wrench.header, rebuild_wrench)
        # print(rebuild_wrenchStamped)
        return rebuild_wrenchStamped

    def convert_my_shit(self, picker):
        convstampF = geometry_msgs.msg.Vector3Stamped(header=self.wrench.header, vector=self.wrench.wrench.force)
        convstampT = geometry_msgs.msg.Vector3Stamped(header=self.wrench.header, vector=self.wrench.wrench.torque)
        # Instead of picker define an array that consists of the 6 values
        if picker == 1:
            self.trans_F_x.append(convstampF.vector.x)
            return self.trans_F_x

        elif picker == 2:
            self.trans_F_y.append(convstampF.vector.y)
            return self.trans_F_y

        elif picker == 3:
            self.trans_F_z.append(convstampF.vector.z)
            return self.trans_F_z

        elif picker == 4:
            self.trans_T_x.append(convstampT.vector.x)
            return self.trans_T_x

        elif picker == 5:
            self.trans_T_y.append(convstampT.vector.y)
            return self.trans_T_y

        elif picker == 6:
            self.trans_T_z.append(convstampT.vector.z)
            return self.trans_T_z

    def butter_lowpass(self, cutoff, fs, order=5):
        nyq = 0.5 * fs  # Nyquist Frequency
        normal_cutoff = cutoff / nyq
        return butter(order, normal_cutoff, btype='low', analog=False)

    def butter_lowpass_filter(self, data, cutoff, fs, order=5):
        b, a = self.butter_lowpass(cutoff, fs, order=order)
        y = lfilter(b, a, data)
        return y


if __name__ == '__main__':
    rospy.init_node('force_torque_raw_filter')
    force = ForceTorqueRawFilter()
    rospy.spin()

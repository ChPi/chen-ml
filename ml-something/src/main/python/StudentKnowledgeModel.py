import tensorflow as tf
import numpy as np


class KnowledgeData(object):
    def __init__(self):
        self.data_file = "data"
        self.student_map = {}
        self.knowledge_map = {}
        self.student_size = 2
        self.knowledge_size = 3
        self.x_student, self.x_knowledge, self.y_rate = self.load_data()

    def load_data(self):
        student_list = []
        knowledge_list = []
        rate_list = []

        # student a--kn1,kn2--rate1,rate2
        with open(self.data_file) as f:
            for i in f:
                student, knowledge_str, rate = i.strip().split("--", 2)
                student = int(student)
                knowledges = list(map(int, knowledge_str.split(",")))
                print(knowledges)
                rates = list(map(float, rate.split(",")))

                knowledge_array = np.zeros(self.knowledge_size)
                knowledge_array[knowledges] = 1
                knowledge_list.append(knowledge_array.tolist())

                student_array = np.zeros(self.student_size)
                student_array[student] = 1
                student_list.append(student_array.tolist())

                rate_array = np.zeros(self.knowledge_size)
                rate_array[knowledges] = rates
                rate_list.append(rate_array.tolist())

        student_x = np.array(student_list)
        knowledge_x = np.array(knowledge_list)
        rate_y = np.array(rate_list)

        return [student_x, knowledge_x, rate_y]


class KnowledgeModel(object):
    def __init__(self, knowledge_size, student_sie, bias=False):
        self.sess = tf.Session()
        self.x = tf.placeholder("float32", shape=[None, knowledge_size])
        self.y = tf.placeholder("float32", shape=[None, knowledge_size])
        self.student = tf.placeholder("float32", shape=[None, student_sie])
        self.bias = bias
        self.coefficient = self.init_variable([knowledge_size, knowledge_size])
        # self.coefficient = tf.tanh(self.coefficient_)
        self.student_knowledge = self.init_variable([student_sie, knowledge_size]) * 10
        self.student_knowledge_real = 1 / (1 + tf.exp(self.student_knowledge))
        self.sess.run(tf.global_variables_initializer())
        # self.coefficient = tf.Print(self.coefficient, [self.x, self.coefficient, self.student_knowledge_real])

    @staticmethod
    def init_variable(shape):
        initial = tf.abs(tf.truncated_normal(shape, stddev=0.1))
        return tf.Variable(initial)

    @property
    def x_ne_zeros(self):
        zero = tf.constant(0, dtype=tf.float32)
        where = tf.not_equal(self.x, zero)
        return tf.where(where)

    @property
    def predict(self):
        w = self.coefficient.eval(session=self.sess)
        np.fill_diagonal(w, 1)
        self.coefficient.assign(w).eval(session=self.sess)
        # print("coefficient------------------------")
        # print(self.coefficient.eval(session=self.sess))
        res1 = tf.matmul(self.x, self.coefficient)
        # res2 = tf.matmul(self.student_knowledge_real, tf.transpose(self.student))
        # print("knowledge------------------------")
        # print(self.student_knowledge_real.eval(session=self.sess))
        res2 = tf.matmul(self.student, self.student_knowledge_real)
        res3 = res1 * res2
        y_ = res3 * self.x
        return y_

    def fit(self, x, student, y):
        print(x)
        print("student------")
        print(student)
        y_ = self.predict
        # cross_entropy = -tf.reduce_sum(self.y * tf.log(y_))
        # cross_entropy = tf.reduce_sum(tf.pow(1 - (self.y / y_), 2))
        cross_entropy = tf.reduce_sum(tf.pow(self.y - y_, 2))
        train_step = tf.train.AdamOptimizer(0.1).minimize(cross_entropy)
        # train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)
        self.sess.run(tf.global_variables_initializer())
        for i in range(1000):
            train_step.run(session=self.sess, feed_dict={self.x: x, self.y: y, self.student: student})
            accuracy = tf.reduce_mean(tf.pow(y_ - y, 2))
            if 0 == i % 50:
                print("coefficient------------------------")
                w = self.coefficient.eval(session=self.sess)
                np.fill_diagonal(w, 1)
                print(w)
                print("predict------------------------")
                print(self.predict.eval(session=self.sess, feed_dict={self.x: x, self.y: y, self.student: student}))
                print("knowledge------------------------")
                print(self.student_knowledge_real.eval(session=self.sess, feed_dict={self.x: x, self.y: y, self.student: student}))
                print("accuracy------------------------")
                print(accuracy.eval(session=self.sess, feed_dict={self.x: x, self.y: y, self.student: student}))


if __name__ == '__main__':
    knowledge_data = KnowledgeData()
    model = KnowledgeModel(knowledge_data.knowledge_size, knowledge_data.student_size)
    model.fit(knowledge_data.x_knowledge, knowledge_data.x_student, knowledge_data.y_rate)




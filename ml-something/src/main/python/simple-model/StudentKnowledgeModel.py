import tensorflow as tf
import numpy as np


class KnowledgeData(object):
    def __init__(self):
        self.data_file = "data"
        self.student_map = {}
        self.student_index_map = {}
        self.knowledge_map = {}
        self.knowledge_index_map = {}
        self.student_index = 0
        self.knowledge_index = 0
        self.knowledge_size = 1409
        self.student_size = 6367
        self.x_student, self.x_knowledge, self.y_rate = self.load_data()

    def add_student_map_(self, student, index):
        self.student_map[student] = index
        self.student_index_map[index] = student

    def add_knowledge_map_(self, knowledge, index):
        self.knowledge_map[knowledge] = index
        self.knowledge_index_map[index] = knowledge

    def student2index(self, student):
        if student in self.student_map:
            student_ = self.student_map[student]
        else:
            student_ = self.student_index
            self.add_student_map_(student, student_)
            self.student_index += 1
        return student_

    def knowledge2index(self, knowledge):
        if knowledge in self.knowledge_map:
            knowledge_ = self.knowledge_map[knowledge]
        else:
            knowledge_ = self.knowledge_index
            self.add_knowledge_map_(knowledge, knowledge_)
            self.knowledge_index += 1
        return knowledge_

    def save_map(self, student_map_file, knowledge_map):
        with open(student_map_file, "w") as f:
            for i, j in self.student_index_map.items():
                f.write(str(i) + "\t" + str(j) + "\n")
        with open(knowledge_map, "w") as f:
            for i, j in self.knowledge_index_map.items():
                f.write(str(i) + "\t" + str(j) + "\n")

    def load_data(self):
        student_list = []
        knowledge_list = []
        rate_list = []

        # student a--kn1,kn2--rate1,rate2
        with open(self.data_file) as f:
            for i in f:
                student_, knowledge_str, rate = i.strip().split("--", 2)
                student = self.student2index(student_)
                print("student size: " + str(self.student_size))
                knowledges = list(map(self.knowledge2index, knowledge_str.split(",")))
                print("knowledge size:" + str(self.knowledge_size))
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
        # w = self.coefficient.eval(session=self.sess)
        # np.fill_diagonal(w, 1)
        self.coefficient = self.coefficient / tf.diag_part(self.coefficient)
        # self.coefficient.assign(w).eval(session=self.sess)
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

    def save_params(self, coefficient_param_file, knowledge_param_file):
        print("start writing coefficient...")
        self.coefficient = self.coefficient / tf.diag_part(self.coefficient)
        coefficient_param = self.coefficient.eval(session=self.sess)
        coefficient_param = coefficient_param.round(4)
        np.savetxt(coefficient_param_file, coefficient_param, fmt='%10.4f', delimiter=",")
        print("coefficient write done. ")
        print("start writing knowledge param...")
        knowledge_param = self.student_knowledge_real.eval(session=self.sess)
        knowledge_param = knowledge_param.round(4)
        np.savetxt(knowledge_param_file, knowledge_param, fmt='%10.4f', delimiter=",")
        print("knowledge param write done. ")

        pass

    def fit(self, x, student, y):
        print("student------")
        y_ = self.predict
        # cross_entropy = -tf.reduce_sum(self.y * tf.log(y_))
        # cross_entropy = tf.reduce_sum(tf.pow(1 - (self.y / y_), 2))
        cross_entropy = tf.reduce_sum(tf.pow(self.y - y_, 2))
        train_step = tf.train.AdamOptimizer(0.1).minimize(cross_entropy)
        # train_step = tf.train.GradientDescentOptimizer(0.01).minimize(cross_entropy)
        self.sess.run(tf.global_variables_initializer())
        for i in range(400):
            train_step.run(session=self.sess, feed_dict={self.x: x, self.y: y, self.student: student})
            accuracy = tf.reduce_mean(tf.pow(y_ - y, 2))
            if 0 == i % 5:
                print("coefficient------------------------")
                w = self.coefficient.eval(session=self.sess)
                np.fill_diagonal(w, 1)
                print(w)
                print("predict------------------------")
                print(self.predict.eval(session=self.sess, feed_dict={self.x: x, self.y: y, self.student: student}))
                print("knowledge------------------------")
                print(self.student_knowledge_real.eval(session=self.sess, feed_dict={self.x: x, self.y: y, self.student: student}))
                print("iter: %s accuracy------------------------"%(i))
                print(accuracy.eval(session=self.sess, feed_dict={self.x: x, self.y: y, self.student: student}))


if __name__ == '__main__':
    knowledge_data = KnowledgeData()
    knowledge_data.save_map("student_index", "knowledge_index")
    model = KnowledgeModel(knowledge_data.knowledge_size, knowledge_data.student_size)
    model.fit(knowledge_data.x_knowledge, knowledge_data.x_student, knowledge_data.y_rate)
    model.save_params("coefficient_params.csv", "knowledge_params.csv")

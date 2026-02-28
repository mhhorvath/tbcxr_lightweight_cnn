class Distiller(tf_keras.Model):
    def __init__(self,student,teacher):
        super().__init__()
        self.teacher = teacher
        self.teacher.trainable = False
        self.student = student

    def compile(
            self,
            optimizer,
            metrics,
            student_loss_fn,
            distillation_loss_fn,
            alpha= 0.7,
            temperature=10
    ):
        super().compile(optimizer=optimizer,metrics=metrics)
        self.student_loss_fn=student_loss_fn
        self.distillation_loss_fn=distillation_loss_fn
        self.alpha = alpha
        self.temperature=temperature

    def train_step(self,data):
        x,y = data
        teacher_pred = self.teacher(x,training=False)
        with GradientTape() as tape:
            student_pred = self.student(x,training=True)
            student_loss = self.student_loss_fn(y,student_pred)
            distillation_loss = self.distillation_loss_fn(
                tf_keras.activations.softmax(
                    teacher_pred / self.temperature #, axis=1
                ),
                tf_keras.activations.softmax(
                    student_pred / self.temperature #, axis=1
                ),
            ) * (self.temperature**2)
            loss = self.alpha * student_loss + (1-self.alpha) * distillation_loss

            grads = tape.gradient(loss, self.student.trainable_variables)
            self.optimizer.apply_gradients(
                zip(grads,self.student.trainable_variables)
            )

            self.compiled_metrics.update_state(y,tf_keras.activations.softmax(student_pred))

            results={m.name : m.result() for m in self.metrics}
            results["loss"] = loss
            return results

    def call(self,x,training=False):
        return self.student(x,training=training)
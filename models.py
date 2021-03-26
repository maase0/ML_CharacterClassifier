import tensorflow as tf

def get_M7_1(num_classes):
    model = tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(input_shape=(63, 64, 1), kernel_size=3, activation='relu', filters=32, name="conv2D_1_1_input"),
        tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=2, name="maxpool_1"),

        tf.keras.layers.Conv2D(kernel_size=3, activation='relu', filters=128, name="conv2D_2_1"),
        tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=2, name="maxpool_2"),

        tf.keras.layers.Conv2D(kernel_size=3, activation='relu', filters=512, name="conv2D_3_1"),
        tf.keras.layers.Conv2D(kernel_size=3, activation='relu', filters=512, name="conv2D_3_2"),
        tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=2, name="maxpool_3"),

        tf.keras.layers.Flatten(name="flatten_1"),
        
        tf.keras.layers.Dense(4096, name="dense_1"),
        tf.keras.layers.Dense(4096, name="dense_2"),

        tf.keras.layers.Dense(num_classes,  name="dense_final"),

        #set the dtype to float32 for numerical stability
        tf.keras.layers.Softmax(dtype="float32", name="softmax_1_output") 
    ], name="M7_1")

    return model

def get_M7_2(num_classes):
    model = tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(input_shape=(63, 64, 1), kernel_size=3, activation='relu', filters=32, name="conv2D_1_1_input"),
        tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=2, name="maxpool_1"),

        tf.keras.layers.Conv2D(kernel_size=3, activation='relu', filters=128, name="conv2D_2_1"),
        tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=2, name="maxpool_2"),

        tf.keras.layers.Conv2D(kernel_size=3, activation='relu', filters=196, name="conv2D_3_1"),
        tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=2, name="maxpool_3"),

        tf.keras.layers.Conv2D(kernel_size=3, activation='relu', filters=256, name="conv2D_4_1"),
        tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=2, name="maxpool_4"),

        tf.keras.layers.Flatten(name="flatten_1"),
        
        
        tf.keras.layers.Dense(1024, name="dense_1"),
        tf.keras.layers.Dense(1024, name="dense_2"),

        tf.keras.layers.Dense(num_classes,  name="dense_final"),

        #set the dtype to float32 for numerical stability
        tf.keras.layers.Softmax(dtype="float32", name="softmax_1_output") 
    ], name="M7_2")

    return model

def get_M9(num_classes):
    model = tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(input_shape=(63, 64, 1), kernel_size=3, activation='relu', filters=32, name="conv2D_1_1_input"),
        tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=2, name="maxpool_1"),

        tf.keras.layers.Conv2D(kernel_size=3, activation='relu', filters=128, name="conv2D_2_1"),
        tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=2, name="maxpool_2"),

        tf.keras.layers.Conv2D(kernel_size=3, activation='relu', filters=256, name="conv2D_3_1"),
        tf.keras.layers.Conv2D(kernel_size=3, activation='relu', filters=256, name="conv2D_3_2"),
        tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=2, name="maxpool_3"),

        tf.keras.layers.Conv2D(kernel_size=3, activation='relu', filters=512, name="conv2D_4_1"),
        tf.keras.layers.Conv2D(kernel_size=3, activation='relu', filters=512, name="conv2D_4_2"),
        tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=2, name="maxpool_4"),

        tf.keras.layers.Flatten(name="flatten_1"),
        
        
        tf.keras.layers.Dense(4096, name="dense_1"),
        tf.keras.layers.Dense(4096, name="dense_2"),

        tf.keras.layers.Dense(num_classes,  name="dense_final"),

        #set the dtype to float32 for numerical stability
        tf.keras.layers.Softmax(dtype="float32", name="softmax_1_output") 
    ], name="M9")

    return model




def get_custom(num_classes):
    model = tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(input_shape=(63, 64, 1), kernel_size=3, activation='relu', filters=32, name="conv2D_1_2_input"),
        tf.keras.layers.Conv2D(kernel_size=3, activation='relu', filters=32, name="conv2D_1_1"),
        tf.keras.layers.Conv2D(kernel_size=3, activation='relu', filters=32, name="conv2D_1_2"),
        tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=2, name="maxpool_1"),

        tf.keras.layers.Conv2D(kernel_size=3, activation='relu', filters=32, name="conv2D_2_1"),
        tf.keras.layers.Conv2D(kernel_size=3, activation='relu', filters=32, name="conv2D_2_2"),

        tf.keras.layers.Conv2D(kernel_size=3, activation='relu', filters=64, name="conv2D_3_1"),
        tf.keras.layers.Conv2D(kernel_size=3, activation='relu', filters=64, name="conv2D_3_2"),
        tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=2, name="maxpool_3"),

        tf.keras.layers.Conv2D(kernel_size=3, activation='relu', filters=128, name="conv2D_4_1"),
        tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=2, name="maxpool_4"),

        tf.keras.layers.Flatten(name="flatten_1"),
        tf.keras.layers.Dropout(0.25, name="dropout_1"),
        
        tf.keras.layers.Dense(2048, name="dense_1"),
        tf.keras.layers.Dropout(0.1, name="dropout_2"),
        tf.keras.layers.Dense(2048, name="densfasave_1"),
        tf.keras.layers.Dropout(0.1, name="dropourstat_2"),

        tf.keras.layers.Dense(2048, name="dense_2"),
        tf.keras.layers.Dropout(0.25, name="dropout_3"),
        tf.keras.layers.Dense(2048, name="densearst_2"),
        tf.keras.layers.Dropout(0.25, name="dropouart_3"),

        tf.keras.layers.Dense(num_classes,  name="dense_3"),

        #set the dtype to float32 for numerical stability
        tf.keras.layers.Softmax(dtype="float32", name="softmax_1_output") 
    ], name="custom_model")

    return model
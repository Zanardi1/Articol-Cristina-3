from dlib_alg import DlibClassification
from facenet_torch import FaceNetTorchClassification
from resnet import TensorFlowClassification
from sk import SKLearnClassification
from vgg_deepface import VGGClassification

print('Incepem cu SKLearn\n')

sk_class = SKLearnClassification()
sk_class.execute()

print('Incepem cu TensorFlow\n')

tf_class = TensorFlowClassification()
tf_class.execute()

print('Incepem cu FaceNet\n')

fn_class = FaceNetTorchClassification()
fn_class.execute()

print('Incepem cu VGG\n')

vgg_class = VGGClassification()
vgg_class.execute()

print('Incepem cu Dlib\n')

dlib_class = DlibClassification()
dlib_class.execute()

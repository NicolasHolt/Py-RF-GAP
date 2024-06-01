from PY_RF_GAP.GAPClassifier import GAPClassifier

def test_GAPClassifier_instance():
    classifier = GAPClassifier()
    assert isinstance(classifier, GAPClassifier), "Object is not an instance of GAPClassifier"
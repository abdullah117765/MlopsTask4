from main import mlops

mlopsObj = mlops(5)
def test_getTotalStudents():
    assert mlopsObj.getTotalStudents() == 5
def test_addStudent():
    mlopsObj.addStudent()
    assert mlopsObj.getTotalStudents() == 6

def test_removeStudent():   
    mlopsObj.removeStudent()
    assert mlopsObj.getTotalStudents() == 5

def test_getClassName():
    assert mlopsObj.getClassName() == "Machine Learning Operations (CS-B)"
import pytest
import gpaw.gpu


@pytest.fixture(scope='session')
def gpu():
    try:
        gpaw.gpu.setup()
    except AttributeError:
        pytest.skip('Not compiled with gpu=True')
    if gpaw.gpu.cupy_is_fake:
        pytest.skip('No cupy')

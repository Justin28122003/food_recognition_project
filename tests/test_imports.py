def test_imports():
    """
    Verifies that the core components are accessible through the package
    structure as defined in the __init__.py files.
    """
    # Test Data module exports
    from ml_core.data import PCAMDataset, get_dataloaders

    assert PCAMDataset is not None
    assert get_dataloaders is not None

    # Test Models module exports
    from ml_core.models import MLP

    assert MLP is not None

    # Test Solver module exports
    from ml_core.solver import Trainer

    assert Trainer is not None

    # Test Utils module exports
    from ml_core.utils import (
        ExperimentTracker,
        load_config,
        seed_everything,
        setup_logger,
    )

    assert ExperimentTracker is not None
    assert load_config is not None
    assert seed_everything is not None
    assert setup_logger is not None

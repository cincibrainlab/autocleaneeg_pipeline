"""Unit tests for mixin discovery system."""

from unittest.mock import Mock, patch

import pytest


# Import will be mocked for tests that don't need full functionality
try:
    from autoclean.mixins import (
        _BASE_MIXIN_CLASS,
        DISCOVERED_MIXINS,
        _base_mixin_found,
        _discovered_other_mixins,
        _warn_on_method_collisions,
    )
    from autoclean.mixins.base import BaseMixin

    MIXINS_AVAILABLE = True
except ImportError:
    MIXINS_AVAILABLE = False
    DISCOVERED_MIXINS = None
    _BASE_MIXIN_CLASS = None
    _discovered_other_mixins = None
    _warn_on_method_collisions = None
    _base_mixin_found = None
    BaseMixin = None


@pytest.mark.skipif(not MIXINS_AVAILABLE, reason="Mixins module not available")
class TestMixinDiscovery:
    """Test the mixin discovery system."""

    def test_discovered_mixins_is_tuple(self):
        """Test that DISCOVERED_MIXINS is a tuple."""

        assert isinstance(DISCOVERED_MIXINS, tuple)
        assert len(DISCOVERED_MIXINS) > 0  # Should have at least BaseMixin

    def test_base_mixin_included(self):
        """Test that BaseMixin functionality is included in discovered mixins."""
        # DISCOVERED_MIXINS already imported at module level, BaseMixin

        # DISCOVERED_MIXINS contains the combined mixin class, not individual mixins
        assert len(DISCOVERED_MIXINS) == 1
        combined_mixin = DISCOVERED_MIXINS[0]

        # The combined mixin should inherit from BaseMixin
        assert issubclass(combined_mixin, BaseMixin)

    def test_base_mixin_class_availability(self):
        """Test that _BASE_MIXIN_CLASS is properly set."""
        # _BASE_MIXIN_CLASS and BaseMixin already imported at module level

        assert _BASE_MIXIN_CLASS == BaseMixin
        assert issubclass(_BASE_MIXIN_CLASS, object)

    def test_discovered_mixins_are_classes(self):
        """Test that all discovered mixins are actual classes."""
        # DISCOVERED_MIXINS already imported at module level

        for mixin in DISCOVERED_MIXINS:
            assert isinstance(mixin, type), f"{mixin} is not a class"
            assert hasattr(mixin, "__name__"), f"{mixin} has no __name__"
            assert hasattr(mixin, "__module__"), f"{mixin} has no __module__"

    def test_mixin_naming_convention(self):
        """Test that discovered mixins follow naming convention."""
        # DISCOVERED_MIXINS already imported at module level

        for mixin in DISCOVERED_MIXINS:
            # Should end with 'Mixin' or 'Mixins' (for CombinedAutocleanMixins)
            assert mixin.__name__.endswith("Mixin") or mixin.__name__.endswith(
                "Mixins"
            ), f"Mixin {mixin.__name__} doesn't follow naming convention"

    def test_mixin_modules_structure(self):
        """Test that mixins come from expected module structure."""
        # DISCOVERED_MIXINS already imported at module level

        for mixin in DISCOVERED_MIXINS:
            module_name = mixin.__module__
            # Should be from autoclean.mixins or its submodules
            assert module_name.startswith(
                "autoclean.mixins"
            ), f"Mixin {mixin.__name__} from unexpected module: {module_name}"


@pytest.mark.skipif(not MIXINS_AVAILABLE, reason="Mixins module not available")
class TestMixinCollisionDetection:
    """Test the mixin method collision detection system."""

    def test_warn_on_method_collisions_no_collisions(self):
        """Test collision detection with no collisions."""
        # _warn_on_method_collisions already imported at module level

        # Create test mixins with no collisions
        class MixinA:
            def method_a(self):
                pass

        class MixinB:
            def method_b(self):
                pass

        # Should not print warnings for no collisions
        with patch("builtins.print") as mock_print:
            _warn_on_method_collisions((MixinA, MixinB))
            # Should not print collision warnings
            collision_warnings = [
                call for call in mock_print.call_args_list if "WARNING:" in str(call)
            ]
            assert len(collision_warnings) == 0

    def test_warn_on_method_collisions_with_collisions(self):
        """Test collision detection with actual collisions."""
        # _warn_on_method_collisions already imported at module level

        # Create test mixins with collisions
        class MixinA:
            def shared_method(self):
                pass

            def unique_a(self):
                pass

        class MixinB:
            def shared_method(self):
                pass

            def unique_b(self):
                pass

        # Should print warnings for collisions
        with patch("builtins.print") as mock_print:
            _warn_on_method_collisions((MixinA, MixinB))

            # Should print collision warning
            print_calls = [str(call) for call in mock_print.call_args_list]
            warning_calls = [call for call in print_calls if "WARNING:" in call]
            assert len(warning_calls) > 0

            # Should mention the conflicting method
            assert any("shared_method" in call for call in warning_calls)

    def test_warn_on_method_collisions_ignores_dunder_methods(self):
        """Test that collision detection ignores dunder methods."""
        # _warn_on_method_collisions already imported at module level

        class MixinA:
            def __init__(self):
                pass

            def __str__(self):
                return "A"

        class MixinB:
            def __init__(self):
                pass

            def __str__(self):
                return "B"

        # Should not warn about dunder method collisions
        with patch("builtins.print") as mock_print:
            _warn_on_method_collisions((MixinA, MixinB))

            print_calls = [str(call) for call in mock_print.call_args_list]
            # Should not mention __init__ or __str__
            assert not any("__init__" in call for call in print_calls)
            assert not any("__str__" in call for call in print_calls)

    def test_method_collision_precedence_detection(self):
        """Test that collision detection shows precedence information."""
        # _warn_on_method_collisions already imported at module level

        class FirstMixin:
            def collision_method(self):
                pass

        class SecondMixin:
            def collision_method(self):
                pass

        with patch("builtins.print") as mock_print:
            _warn_on_method_collisions((FirstMixin, SecondMixin))

            print_calls = [str(call) for call in mock_print.call_args_list]

            # Should mention precedence
            precedence_calls = [
                call for call in print_calls if "appears earliest" in call
            ]
            assert len(precedence_calls) > 0

            # Should mention FirstMixin has precedence
            assert any("FirstMixin" in call for call in precedence_calls)


@pytest.mark.skipif(not MIXINS_AVAILABLE, reason="Mixins module not available")
class TestBaseMixin:
    """Test the BaseMixin class functionality."""

    def test_base_mixin_importable(self):
        """Test that BaseMixin can be imported."""
        # BaseMixin already imported at module level

        assert BaseMixin is not None
        assert isinstance(BaseMixin, type)

    def test_base_mixin_has_expected_interface(self):
        """Test that BaseMixin has expected interface."""
        # BaseMixin already imported at module level

        # Should be a class that can be inherited from
        class TestClass(BaseMixin):
            pass

        instance = TestClass()
        assert isinstance(instance, BaseMixin)

    def test_base_mixin_in_discovered_mixins(self):
        """Test that BaseMixin functionality is properly included in discovery."""
        # DISCOVERED_MIXINS already imported at module level
        # BaseMixin already imported at module level

        # The combined mixin should inherit from BaseMixin
        combined_mixin = DISCOVERED_MIXINS[0]
        assert issubclass(combined_mixin, BaseMixin)






# Error handling and edge cases
class TestMixinDiscoveryEdgeCases:
    """Test mixin discovery edge cases and error conditions."""

    @pytest.mark.skipif(not MIXINS_AVAILABLE, reason="Mixins module not available")
    def test_base_mixin_was_found_during_discovery(self):
        """BaseMixin should always be located by the discovery system in a healthy install."""
        assert _base_mixin_found is True

    @pytest.mark.skipif(not MIXINS_AVAILABLE, reason="Mixins module not available")
    def test_collision_detection_ignores_empty_mixins(self):
        """_warn_on_method_collisions should not warn for mixins with no methods."""

        class EmptyMixin:
            pass

        with patch("builtins.print") as mock_print:
            _warn_on_method_collisions((EmptyMixin,))
            collision_warnings = [
                call for call in mock_print.call_args_list if "WARNING:" in str(call)
            ]
            assert len(collision_warnings) == 0

    @pytest.mark.skipif(not MIXINS_AVAILABLE, reason="Mixins module not available")
    def test_collision_detection_ignores_non_callable_attributes(self):
        """_warn_on_method_collisions should not treat class variables as collision candidates."""

        class MixinA:
            class_variable = "test"

            def real_method(self):
                pass

        class MixinB:
            class_variable = "other"

            def other_method(self):
                pass

        with patch("builtins.print") as mock_print:
            _warn_on_method_collisions((MixinA, MixinB))
            collision_warnings = [
                call for call in mock_print.call_args_list if "WARNING:" in str(call)
            ]
            # class_variable should not trigger a collision warning
            assert not any("class_variable" in str(c) for c in collision_warnings)

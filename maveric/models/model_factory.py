"""Factory for creating model instances."""

from typing import Dict, Type, Optional, Any
from ..core.exceptions import ModelError


class ModelFactory:
    """Factory for creating model wrappers."""
    
    _models: Dict[str, Type] = {}
    
    @classmethod
    def register(cls, name: str, model_class: Type):
        """Register a model class."""
        cls._models[name.lower()] = model_class
    
    @classmethod
    def create_model(cls, model_type: str, **kwargs) -> Any:
        """
        Create a model instance.
        
        Args:
            model_type: Type of model to create
            **kwargs: Model-specific arguments
            
        Returns:
            Model instance
        """
        model_type_lower = model_type.lower()
        
        if model_type_lower not in cls._models:
            available = ', '.join(sorted(cls._models.keys()))
            raise ModelError(
                f"Model type '{model_type}' not supported. "
                f"Available types: {available}"
            )
        
        model_class = cls._models[model_type_lower]
        return model_class(**kwargs)


def get_model(model_type: str, **kwargs) -> Any:
    """
    Convenience function to get a model.
    
    Args:
        model_type: Type of model
        **kwargs: Model arguments
        
    Returns:
        Model instance
    """
    return ModelFactory.create_model(model_type, **kwargs)


# Register built-in models
def _register_builtin_models():
    """Register built-in model types."""
    from .clip_wrapper import CLIPWrapper
    
    ModelFactory.register('clip', CLIPWrapper)
    ModelFactory.register('clip-vit-b-32', CLIPWrapper)


# Register on import
_register_builtin_models()

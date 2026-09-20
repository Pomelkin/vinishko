import importlib
from typing import Self


if importlib.util.find_spec("transformers") is not None:
    from transformers import AutoProcessor
    from transformers import AutoTokenizer
    from transformers import SentencePieceBackend
    from transformers import TokenizersBackend

    class AllPurposeWrapper:
        def __new__[T](cls, class_to_instanciate: T, *args, **kwargs) -> Self:
            return class_to_instanciate.from_pretrained(*args, **kwargs)  # ty: ignore[unresolved-attribute]

    class AutoProcessorWrapper:
        def __new__[T: AutoProcessor](cls: type[T], *args, **kwargs) -> T:
            return AutoProcessor.from_pretrained(*args, **kwargs)

    class AutoTokenizerWrapper:
        def __new__[T: SentencePieceBackend | TokenizersBackend](
            cls: type[T], *args, **kwargs
        ) -> T:
            return AutoTokenizer.from_pretrained(*args, **kwargs)  # ty: ignore[invalid-return-type]

else:
    raise ModuleNotFoundError("Transformers must be loaded")

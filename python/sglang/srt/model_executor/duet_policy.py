"""Conservative cache identity for the eager final DUET release path."""
import uuid


def release_config(model_config):
    config = model_config.hf_config
    value = getattr(config, "twinstar", None)
    if value is None:
        value = getattr(getattr(config, "text_config", None), "twinstar", None)
    return value if isinstance(value, dict) and value.get("duet_release") else None


def namespace_request(model_config, req):
    """Only an explicit session can reuse D states across HTTP requests.

    DUET P and D are distinct computations. Token equality alone does not make
    a completed D prefix a valid P state for a new cold prompt. A fresh nonce
    also isolates two calls that reuse the same user-provided request id.
    """
    config = release_config(model_config)
    if config is None:
        return
    session = getattr(req, "session_id", None)
    if session is None and getattr(req, "session", None) is not None:
        session = req.session.session_id
    if session is not None and (not isinstance(session, str) or not session):
        raise ValueError("DUET session_id must be a nonempty string")
    # JSON length-delimits user data, avoiding separator collisions.
    import json
    namespace = json.dumps(["duet-v1", config["duet_release"],
                            config.get("duet_sha256"),
                            "session" if session is not None else "cold",
                            session if session is not None else uuid.uuid4().hex,
                            req.extra_key], separators=(",", ":"))
    req.extra_key = namespace

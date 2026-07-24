from cmfgpu.phys import cuda


def test_precompile_for_opened_modules_builds_exact_demand(monkeypatch) -> None:
    calls = []
    monkeypatch.setattr(
        cuda._CUDA, "ensure_precompiled",
        lambda extensions: calls.append(set(extensions)) or {},
    )
    cuda._CUDA.ensure_precompiled_for_modules(
        ("base", "adaptive_time", "bifurcation"),
    )
    assert calls == [{"storage", "outflow", "adaptive", "bifurcation"}]


def test_precompile_for_optional_modules_adds_only_their_extensions(
    monkeypatch,
) -> None:
    calls = []
    monkeypatch.setattr(
        cuda._CUDA, "ensure_precompiled",
        lambda extensions: calls.append(set(extensions)) or {},
    )
    cuda._CUDA.ensure_precompiled_for_modules(("base", "reservoir", "levee"))
    assert calls == [{"storage", "outflow", "reservoir", "levee"}]

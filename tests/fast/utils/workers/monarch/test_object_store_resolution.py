from argparse import Namespace

import pytest

from miles.utils.arguments import _resolve_object_store_of_cluster


def _args(*, cluster_backend: str, object_store_backend: str = "ray") -> Namespace:
    return Namespace(
        cluster_backend=cluster_backend,
        object_store_backend=object_store_backend,
        mooncake_store_init_kwargs=None,
        deploy_component="all",
    )


class TestResolveObjectStoreOfCluster:
    @pytest.mark.parametrize(
        ("cluster_backend", "resolved"), [("ray", "ray"), ("kubernetes", "mooncake"), ("monarch", "monarch_rdma")]
    )
    def test_the_default_follows_the_cluster(self, cluster_backend: str, resolved: str):
        """Without a Ray cluster there is no Ray object store; monarch hands data over as its own rdma buffers."""
        args = _args(cluster_backend=cluster_backend)

        _resolve_object_store_of_cluster(args)

        assert args.object_store_backend == resolved

    def test_monarch_can_still_ask_for_mooncake(self):
        args = _args(cluster_backend="monarch", object_store_backend="mooncake")

        _resolve_object_store_of_cluster(args)

        assert args.object_store_backend == "mooncake"
        assert args.mooncake_store_init_kwargs is not None

    @pytest.mark.parametrize("cluster_backend", ["ray", "kubernetes"])
    def test_rdma_buffers_need_monarch_processes(self, cluster_backend: str):
        with pytest.raises(AssertionError, match="only processes launched by --cluster-backend monarch"):
            _resolve_object_store_of_cluster(
                _args(cluster_backend=cluster_backend, object_store_backend="monarch_rdma")
            )

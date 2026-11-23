# Copyright 2022 the Regents of the University of California, Nerfstudio Team and contributors. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""CPU checkpoint ownership contract, without importing renderer/GPU dependencies.

Execute the actual Trainer checkpoint method and Splatfacto parameter loader
from their source AST. The surrounding setup/console/checkpoint I/O is a fixture;
the methods under test are neither copied nor reimplemented. A separate installed
Splatfacto CPU probe reproduced the same stale-optimizer binding (attack#84 D2).
"""

import ast
import copy
import os
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from torch import nn

ROOT = Path(__file__).resolve().parents[2]


def source_method_class(relative, name, method, base, namespace):
    """Compile one unchanged production method with its real __class__ cell."""
    tree = ast.parse((ROOT / relative).read_text())
    cls = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == name)
    fn = next(n for n in cls.body if isinstance(n, ast.FunctionDef) and n.name == method)
    selected = ast.ClassDef(name=name, bases=[ast.Name(id="FixtureBase", ctx=ast.Load())],
                            keywords=[], body=[copy.deepcopy(fn)], decorator_list=[])
    module = ast.fix_missing_locations(ast.Module(body=[selected], type_ignores=[]))
    scope = {"FixtureBase": base, **namespace}
    exec(compile(module, str(ROOT / relative), "exec"), scope)
    return scope[name]


class CPUModel(nn.Module):
    @property
    def device(self):
        return torch.device("cpu")


class OptimizerFixture:
    def __init__(self, model):
        self.optimizer = torch.optim.Adam([model.gauss_params["means"]], lr=0.01)
        self.schedulers = None

    def load_optimizers(self, state):
        self.optimizer.load_state_dict(state["means"])

    def load_schedulers(self, state):
        self.schedulers = state


def fixture(tmp_path, mode, count, scheduler=True):
    model_type = source_method_class("nerfstudio/models/splatfacto.py", "SplatfactoModel", "load_state_dict",
                                     CPUModel, {"torch": torch})
    model = model_type()
    model.gauss_params = nn.ParameterDict({"means": nn.Parameter(torch.zeros((3, 3)))})
    saved = {"gauss_params.means": torch.ones((count, 3))}
    saved_model = model_type()
    saved_model.gauss_params = nn.ParameterDict({"means": nn.Parameter(torch.ones((count, 3)))})
    saved_optimizer = OptimizerFixture(saved_model)
    saved_model.gauss_params["means"].sum().backward()
    saved_optimizer.optimizer.step()
    saved["gauss_params.means"] = saved_model.gauss_params["means"].detach().clone()
    state = {"step": 7, "pipeline": saved, "optimizers": {"means": saved_optimizer.optimizer.state_dict()},
             "schedulers": {"means": {"last_epoch": 7}}, "scalers": {"scale": 42}}
    checkpoint = tmp_path / "step-000000007.ckpt"
    torch.save(state, checkpoint)
    loads = []

    def read(path):
        loads.append(path)
        return torch.load(path, map_location="cpu", weights_only=True)

    trainer_type = source_method_class("nerfstudio/engine/trainer.py", "Trainer", "_load_checkpoint", object,
                                       {"Path": Path, "os": os, "load_checkpoint_state": read,
                                        "CONSOLE": SimpleNamespace(print=lambda *_: None)})
    trainer = trainer_type()
    trainer.config = SimpleNamespace(load_dir=tmp_path if mode.startswith("dir") else None,
                                    load_checkpoint=checkpoint if mode == "file" else None,
                                    load_step=7 if mode == "dir-step" else None, load_scheduler=scheduler)
    trainer.pipeline = SimpleNamespace(load_pipeline=lambda state, _step: model.load_state_dict(state))
    trainer.optimizers = OptimizerFixture(model)
    trainer.rebuilds = []

    def setup():
        result = OptimizerFixture(model)
        trainer.rebuilds.append(result)
        return result

    trainer.setup_optimizers = setup
    trainer.grad_scaler = SimpleNamespace(load_state_dict=lambda state: setattr(trainer, "scaler_state", state))
    return trainer, model, checkpoint, loads


@pytest.mark.parametrize("mode", ["file", "dir-latest", "dir-step"])
@pytest.mark.parametrize("count", [3, 5])
@pytest.mark.parametrize("scheduler", [False, True])
def test_resume_updates_loaded_parameters_and_restores_optimizer_state(tmp_path, mode, count, scheduler):
    trainer, model, checkpoint, loads = fixture(tmp_path, mode, count, scheduler)
    original = model.gauss_params["means"]
    trainer._load_checkpoint()
    current = model.gauss_params["means"]
    assert current is not original and current.shape == (count, 3)
    assert len(trainer.rebuilds) == 1
    optimizer = trainer.optimizers.optimizer
    assert optimizer.param_groups[0]["params"][0] is current
    assert optimizer.state[current]["step"].item() == 1
    assert optimizer.state[current]["exp_avg"].shape == current.shape
    assert trainer._start_step == 8 and loads == [checkpoint]
    assert trainer.scaler_state == {"scale": 42}
    assert (trainer.optimizers.schedulers is not None) == scheduler
    optimizer.zero_grad()
    before = current.detach().clone()
    current.sum().backward()
    optimizer.step()
    assert not torch.equal(current, before), "resumed optimization must update the loaded model"
    assert not torch.cuda.is_initialized()


def test_fresh_training_keeps_its_existing_optimizer(tmp_path):
    trainer, model, _checkpoint, loads = fixture(tmp_path, "fresh", 3)
    original = trainer.optimizers
    trainer._load_checkpoint()
    assert trainer.optimizers is original and not trainer.rebuilds and not loads
    assert original.optimizer.param_groups[0]["params"][0] is model.gauss_params["means"]

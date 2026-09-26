"""ProductPlan + DesignSpec -> ordered, dependency-aware TaskGraph.

Kept deterministic/rule-based rather than model-driven where possible:
task decomposition from a plan is largely mechanical (one task per
entity/API/page/component), which makes builds reproducible and cheap.
"""
from __future__ import annotations

from planning.schemas import ProductPlan, Task, TaskGraph


def build_task_graph(plan: ProductPlan) -> TaskGraph:
    tasks: list[Task] = []

    scaffold = Task(id="scaffold", title="Scaffold project", description="Initialize app skeleton")
    tasks.append(scaffold)

    entity_ids = []
    for entity in plan.entities:
        tid = f"entity:{entity.name}"
        tasks.append(Task(
            id=tid,
            title=f"Define {entity.name} schema",
            description=f"Create data model for {entity.name} with fields {entity.fields}",
            depends_on=[scaffold.id],
        ))
        entity_ids.append(tid)

    api_ids = []
    for endpoint in plan.api_endpoints:
        tid = f"api:{endpoint}"
        tasks.append(Task(
            id=tid,
            title=f"Implement {endpoint}",
            description=f"Implement API endpoint {endpoint}",
            depends_on=entity_ids or [scaffold.id],
        ))
        api_ids.append(tid)

    if plan.needs_auth:
        auth_task = Task(
            id="auth",
            title="Implement authentication",
            description="Add auth flow (sign up, sign in, session handling)",
            depends_on=[scaffold.id],
        )
        tasks.append(auth_task)
        api_ids.append(auth_task.id)

    for page in plan.pages:
        tid = f"page:{page.route}"
        tasks.append(Task(
            id=tid,
            title=f"Build page {page.name}",
            description=f"Implement {page.route}: {page.purpose}",
            depends_on=api_ids or [scaffold.id],
        ))

    return TaskGraph(tasks=tasks)

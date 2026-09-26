# Scale

This pipeline currently runs on a single workstation (12-16 cores). A separate substrate exists
that could run it at scale; this page is the honest split between what that substrate has
demonstrated and what it has not.

## What is demonstrated

A private repository, **constellaration-fleet**, built a GKE + KubeRay stack for running workloads
shaped like this one at scale, and has demonstrated:

- **GKE + KubeRay**, deployed and live.
- **Scale-to-zero**: the cluster idles at zero cost between runs.
- **Chaos-tested preemption resilience**: the stack has been deliberately tested against
  preemptible-node loss and recovers.
- **An observability gate**: run health is monitored, not just logged.
- **A keyless CD path**: deployment does not depend on long-lived static credentials.

## What is not demonstrated

- **No scientific-scale run.** The stack has not yet run any of this repository's actual physics
  (P1/P2/P3 evaluations, the restoration engine, front-mapping) at any scale beyond what a single
  workstation already does here.
- **The P3 scripts have not been ported.** Nothing in `src/constellaration_methods/` has been
  adapted to run as a KubeRay job; this repository and the fleet substrate are, today, two
  separate things that have not yet been connected.

## The open item

Porting this repository's pipeline onto that substrate -- so a re-optimisation or front-mapping run
becomes a KubeRay job instead of a single-workstation `podman run` -- is the open item. It is the
natural next step if a "definition in, confirmed front out" service (see `QUESTIONS.md`) needs to
run faster than 12 cores can go, but it has not been started.

See constellaration-fleet for the substrate itself (link to be added).

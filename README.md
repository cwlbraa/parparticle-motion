parparticle-motion
==================

CS194 engineering parallel software project

Interfaces:
    Input:
        -> openCV video codec to XXX
    Input to Emission Model:
        -> hashmap<position, probability>
    Particle filter to Video Output:
        -> either list of particle positions or array of particle counts
    Output:
        -> render video w/ particle overlay


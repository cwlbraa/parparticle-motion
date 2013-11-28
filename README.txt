parparticle-motion
==================

CS194 engineering parallel software project

headers go in src, keep libraries and imports in folders inside root

Interfaces:
    Input:
        -> openCV video codec to XXX
    Input to Emission Model:
        -> std::map<position, probability>


    Particle filter to Video Output:
        -> either list of particle positions or array of particle counts
    Output:
        -> render video w/ particle overlay


using System.Collections;
using System.Collections.Generic;
using UnityEngine;

public class SoundFX : MonoBehaviour
{
    public AudioSource laser;
    public AudioSource coli;
    
    public void Playlaser() {
        laser.Play();
    }
    public void Playcoli()
    {
       coli.Play();
    }
}

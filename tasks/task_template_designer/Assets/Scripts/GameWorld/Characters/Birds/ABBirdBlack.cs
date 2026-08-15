using UnityEngine;
using System.Collections;

public class ABBirdBlack : ABBird {

	public float _explosionArea = 1f;
	public float _explosionPower = 1f;
	public float _explosionDamage = 1f;

	void SpecialAttack() {

		Explode ();
	}

	// Called via frame event
	void Explode() {

		ABTNT.Explode (transform.position, _explosionArea, _explosionPower, _explosionDamage, gameObject);
		Die (true);
	}

	public override void OnCollisionEnter2D(Collision2D collision) {

		// This override does not call base, so ABGameObject's recorder call never
		// runs for a BirdBlack. The bird is the one object every shot on the smoke
		// level is guaranteed to reach, so without this the capture has no
		// collision to record at all. Called directly rather than through base:
		// the base handler also runs the damage model, which must stay off the bird.
		PhysicalSnapshotRuntime.RecordCollisionCallback(collision);

		_trailParticles._shootParticles = false;

		if(OutOfSlingShot)
			_animator.Play ("explode");
	}
}

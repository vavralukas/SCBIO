using UnityEngine;
using UnityEngine.SceneManagement;

public class upndown : MonoBehaviour
{
    bool up = false;
    bool down = false;

    Rigidbody rb;
    Transform sujetacam;
    [SerializeField] float veloJuga;
    [SerializeField] float fuerzaMovi;
    Vector3 vec;

    [SerializeField] GameObject obstacle;
    [SerializeField] float obstacleDistance;
    [SerializeField] float obstaclePosY;
    [SerializeField] int obstacleNum;
    AudioSource laser;


    void buildObstacles()
    {
        vec.z = 56.4f;
        for (int i = 0; i < obstacleNum; i++)
        {
            vec.z += obstacleDistance;
            vec.y = Random.Range(-obstaclePosY, obstaclePosY);

            Instantiate(obstacle, vec, Quaternion.identity);
        }
    }

    // Start is called before the first frame update
    void Start()
    {
        rb = GetComponent<Rigidbody>();
        sujetacam = Camera.main.transform.parent;
        laser = GetComponent<AudioSource>();

        buildObstacles();
    }

    // Update is called once per frame
    void Update()
    {
        if (Input.GetKeyUp(KeyCode.UpArrow))
        {
            up = true;
            laser.Play();
            
        }
        if (Input.GetKeyUp(KeyCode.DownArrow))
        {
            down = true;
            laser.Play();
        }
    }
    void FixedUpdate()
    {
        //rb.AddForce(Vector3.forward * veloJuga * Time.fixedDeltaTime);
        transform.Translate(Vector3.forward * veloJuga * Time.fixedDeltaTime);
        if (up)
        {
            transform.Translate(Vector3.up * Time.fixedDeltaTime*fuerzaMovi);
            up = false;
        }
        if (down)
        {
            transform.Translate(Vector3.down * Time.fixedDeltaTime * fuerzaMovi);
            down = false;
        }
        //* Time.fixedDeltaTime
    }
    void LateUpdate()
    {
        vec.x = sujetacam.transform.position.x;
        vec.y = sujetacam.transform.position.y;
        vec.z = transform.position.z;

        sujetacam.transform.position = vec;
    }

    void OnCollisionEnter()
    {
        Invoke ("RestartGame", 1f);
        //Debug.Log("Game Over");
        //Time.timeScale = 0;

    }

    void RestartGame()
    {
        SceneManager.LoadScene(0);
    }
}

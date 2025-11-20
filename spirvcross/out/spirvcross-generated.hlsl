#ifndef SPIRV_CROSS_CONSTANT_ID_0
#define SPIRV_CROSS_CONSTANT_ID_0 1u
#endif
static const uint _16 = SPIRV_CROSS_CONSTANT_ID_0;
#ifndef SPIRV_CROSS_CONSTANT_ID_1
#define SPIRV_CROSS_CONSTANT_ID_1 1u
#endif
static const uint _17 = SPIRV_CROSS_CONSTANT_ID_1;
static const uint3 gl_WorkGroupSize = uint3(_16, _17, 1u);

globallycoherent RWByteAddressBuffer _7 : register(u0, space0);
globallycoherent RWByteAddressBuffer _10 : register(u2, space0);
globallycoherent RWByteAddressBuffer _13 : register(u1, space0);
cbuffer _14_45
{
    int _45_m0 : packoffset(c0);
    int _45_m1 : packoffset(c0.y);
    int _45_m2 : packoffset(c0.z);
    uint _45_m3 : packoffset(c0.w);
    uint _45_m4 : packoffset(c1);
    uint _45_m5 : packoffset(c1.y);
};


static uint gl_LocalInvocationIndex;
struct SPIRV_Cross_Input
{
    uint gl_LocalInvocationIndex : SV_GroupIndex;
};

static int _29;

bool _255(uint4 _345, uint _346)
{
    return ((_345[_346 / 32u] >> (_346 % 32u)) & 1u) != 0u;
}

void comp_main()
{
    _29 = 0;
    if (WaveIsFirstLane())
    {
        for (int _67 = 0; uint(_67) < _7.Load<uint>(16); _67++)
        {
        }
        if (_7.Load<uint>(12) == 3u)
        {
            _10.Store<uint>(gl_LocalInvocationIndex * 4 + 0, _10.Load<uint>(gl_LocalInvocationIndex * 4 + 0) + uint(1));
            int _97 = _29;
            _29 = _97 + 1;
            _13.Store<uint4>((uint(_97 * _45_m0) + gl_LocalInvocationIndex) * 16 + 0, WaveActiveBallot(true));
        }
        else
        {
            _10.Store<uint>(gl_LocalInvocationIndex * 4 + 0, _10.Load<uint>(gl_LocalInvocationIndex * 4 + 0) + uint(1));
            int _111 = _29;
            _29 = _111 + 1;
            _13.Store<uint4>((uint(_111 * _45_m0) + gl_LocalInvocationIndex) * 16 + 0, WaveActiveBallot(true));
        }
    }
    if (_7.Load<uint>(0) == 0u)
    {
        int _68 = 0;
        do
        {
            if (WaveIsFirstLane())
            {
                break;
            }
            _10.Store<uint>(gl_LocalInvocationIndex * 4 + 0, _10.Load<uint>(gl_LocalInvocationIndex * 4 + 0) + uint(1));
            int _137 = _29;
            _29 = _137 + 1;
            _13.Store<uint>((uint(_137 * _45_m0) + gl_LocalInvocationIndex) * 16 + 0, 65550u);
            _68++;
        } while (true);
        if (gl_LocalInvocationIndex >= _7.Load<uint>(152))
        {
            _10.Store<uint>(gl_LocalInvocationIndex * 4 + 0, _10.Load<uint>(gl_LocalInvocationIndex * 4 + 0) + uint(1));
            int _159 = _29;
            _29 = _159 + 1;
            _13.Store<uint>((uint(_159 * _45_m0) + gl_LocalInvocationIndex) * 16 + 0, 65553u);
        }
        else
        {
            _10.Store<uint>(gl_LocalInvocationIndex * 4 + 0, _10.Load<uint>(gl_LocalInvocationIndex * 4 + 0) + uint(1));
            int _172 = _29;
            _29 = _172 + 1;
            _13.Store<uint>((uint(_172 * _45_m0) + gl_LocalInvocationIndex) * 16 + 0, 65555u);
        }
        _10.Store<uint>(gl_LocalInvocationIndex * 4 + 0, _10.Load<uint>(gl_LocalInvocationIndex * 4 + 0) + uint(1));
        int _185 = _29;
        _29 = _185 + 1;
        _13.Store<uint>((uint(_185 * _45_m0) + gl_LocalInvocationIndex) * 16 + 0, 65557u);
    }
    _10.Store<uint>(gl_LocalInvocationIndex * 4 + 0, _10.Load<uint>(gl_LocalInvocationIndex * 4 + 0) + uint(1));
    int _198 = _29;
    _29 = _198 + 1;
    _13.Store<uint4>((uint(_198 * _45_m0) + gl_LocalInvocationIndex) * 16 + 0, WaveActiveBallot(true));
    if (_7.Load<uint>(8) == 2u)
    {
        _10.Store<uint>(gl_LocalInvocationIndex * 4 + 0, _10.Load<uint>(gl_LocalInvocationIndex * 4 + 0) + uint(1));
        int _217 = _29;
        _29 = _217 + 1;
        _13.Store<uint4>((uint(_217 * _45_m0) + gl_LocalInvocationIndex) * 16 + 0, WaveActiveBallot(true));
        _10.Store<uint>(gl_LocalInvocationIndex * 4 + 0, _10.Load<uint>(gl_LocalInvocationIndex * 4 + 0) + uint(1));
        int _231 = _29;
        _29 = _231 + 1;
        _13.Store<uint>((uint(_231 * _45_m0) + gl_LocalInvocationIndex) * 16 + 0, 65562u);
    }
    _10.Store<uint>(gl_LocalInvocationIndex * 4 + 0, _10.Load<uint>(gl_LocalInvocationIndex * 4 + 0) + uint(1));
    int _244 = _29;
    _29 = _244 + 1;
    _13.Store<uint4>((uint(_244 * _45_m0) + gl_LocalInvocationIndex) * 16 + 0, WaveActiveBallot(true));
    uint4 _69 = uint4(3701694078u, 3701687786u, 528435240u, 1390549452u);
    uint _70 = WaveGetLaneIndex();
    if (_255(_69, _70))
    {
        if (gl_LocalInvocationIndex >= _7.Load<uint>(44))
        {
            _10.Store<uint>(gl_LocalInvocationIndex * 4 + 0, _10.Load<uint>(gl_LocalInvocationIndex * 4 + 0) + uint(1));
            int _269 = _29;
            _29 = _269 + 1;
            _13.Store<uint4>((uint(_269 * _45_m0) + gl_LocalInvocationIndex) * 16 + 0, WaveActiveBallot(true));
            _10.Store<uint>(gl_LocalInvocationIndex * 4 + 0, _10.Load<uint>(gl_LocalInvocationIndex * 4 + 0) + uint(1));
            int _283 = _29;
            _29 = _283 + 1;
            _13.Store<uint>((uint(_283 * _45_m0) + gl_LocalInvocationIndex) * 16 + 0, 65568u);
        }
    }
    else
    {
        _10.Store<uint>(gl_LocalInvocationIndex * 4 + 0, _10.Load<uint>(gl_LocalInvocationIndex * 4 + 0) + uint(1));
        int _296 = _29;
        _29 = _296 + 1;
        _13.Store<uint4>((uint(_296 * _45_m0) + gl_LocalInvocationIndex) * 16 + 0, WaveActiveBallot(true));
        for (int _71 = 0; uint(_71) < _7.Load<uint>(8); _71++)
        {
            _10.Store<uint>(gl_LocalInvocationIndex * 4 + 0, _10.Load<uint>(gl_LocalInvocationIndex * 4 + 0) + uint(1));
            int _320 = _29;
            _29 = _320 + 1;
            _13.Store<uint4>((uint(_320 * _45_m0) + gl_LocalInvocationIndex) * 16 + 0, WaveActiveBallot(true));
        }
        _10.Store<uint>(gl_LocalInvocationIndex * 4 + 0, _10.Load<uint>(gl_LocalInvocationIndex * 4 + 0) + uint(1));
        int _336 = _29;
        _29 = _336 + 1;
        _13.Store<uint>((uint(_336 * _45_m0) + gl_LocalInvocationIndex) * 16 + 0, 65575u);
    }
}

[numthreads(SPIRV_CROSS_CONSTANT_ID_0, SPIRV_CROSS_CONSTANT_ID_1, 1)]
void main(SPIRV_Cross_Input stage_input)
{
    gl_LocalInvocationIndex = stage_input.gl_LocalInvocationIndex;
    comp_main();
}
